{-# LANGUAGE NoMonomorphismRestriction,
 RankNTypes,KindSignatures,DataKinds,GADTs,FlexibleContexts
 #-}
module AI.HFNN.Internal (
  WeightSelector,
  Layer,
  NNBuilder,
  NNStructure,
  WeightValues,
  FeedForward,
  WeightUpdate,
  InputTension,
  structureNodes,
  structureBaseWeights,
  structureInputs,
  structureOutputs,
  bias,
  addInputs,
  layerSize,
  addBaseWeights,
  fixedWeights,
  linearLayer,
  standardLayer,
  stochasticLayer,
  pointwiseSum,
  pointwiseProduct,
  pointwiseUnary,
  splitLayer,
  layerNodes,
  concatLayers,
  addOutputs,
  initialWeights,
  initialWeights',
  runNNBuilder,
  serializeWeights,
  deserializeWeights,
  packWeights,
  unpackWeights,
  weightedSumUpdates,
  feedForward,
  getOutput,
  getOutputs,
  backPropagate,
  inputError,
  applyDelta,
  applyDeltaWith
 ) where

import Control.Monad
import Data.Array.IO
import Data.List (foldl1')
import Data.Semigroup
import Data.Word
import qualified Data.ByteString as BS
import qualified Data.ByteString.Internal as BS
import Foreign.ForeignPtr
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.Storable
import System.IO.Unsafe
import System.Random

import AI.HFNN.Activation

-- | Represents the relationship between a linear array of doubles and a
-- particular weight matrix. Phantom type to ensure only directed acyclic
-- graphs are created.
newtype WeightSelector s = WS IWeightSelector

data IWeightSelector = IWeightSelector {
  weightsInputs :: Word,
  weightsOutputs :: Word,
  getWeight :: (Word -> IO Double) -> Word -> Word -> IO Double,
  updateWeight :: (Word -> Double -> IO ()) -> Word -> Word -> Double -> IO ()
 }

wsInputs (WS a) = weightsInputs a
wsOutputs (WS a) = weightsOutputs a

-- | Represents a set of neurons which take inputs from a common set of parents.
-- Phantom type to ensure directed acyclic graphs are generated, and that
-- no arrays are indexed out of bounds.
newtype Layer s = Layer ILayer

data ILayer = ILayer Word Word

-- | Bias node: value is always 1.
bias :: forall s . Layer s
bias = Layer (ILayer 0 0)

data NNOperation (a :: Bool) where
  WeightPatch :: Word -> Word -> IWeightSelector -> NNOperation a
  ApplyActivation :: Word -> Word -> ActivationFunction -> NNOperation a
  ApplyRandomization :: Word -> Word -> (forall g . RandomGen g =>
    g -> Double -> Double -> (Double, Double,g)
   ) -> NNOperation True
  PointwiseSum :: Word -> Word -> [(Word,Word)] -> NNOperation a
  PointwiseProduct :: Word -> Word -> [(Word,Word)] -> NNOperation a
  PointwiseUnary :: Word -> Word -> Word -> (Double -> (Double, Double)) ->
    NNOperation a
  Copy :: Word -> Word -> Word -> NNOperation a

-- | A monad for assembling feedforward networks. The boolean type parameter
-- indicates whether or not the network may use stochastic units
newtype NNBuilder (d :: Bool) s a = NNBuilder (
  Word -> Word ->
  CatTree Word ->
  CatTree Word ->
  CatTree (NNOperation d) ->
  (Word, Word, CatTree Word, CatTree Word, CatTree (NNOperation d), a)
 )

-- | The directed acyclic graph and activation functions making up a neural
-- network. Actual weights are stored in the corresponding 'WeightValues'
-- structures.
data NNStructure (d :: Bool) = NNStructure {
  countNodes :: Word,
  countBaseWeights :: Word,
  inputNodes :: IOUArray Word Word,
  outputNodes :: IOUArray Word Word,
  nnOperations :: IOArray Word (NNOperation d)
 }

-- | The total number of nodes in the neural network
structureNodes :: NNStructure d -> Word
structureNodes = countNodes

-- | The number of weight variables in the neural network
structureBaseWeights :: NNStructure d -> Word
structureBaseWeights = countBaseWeights

structureInputs :: NNStructure d -> Word
structureInputs ns = unsafePerformIO $ (\(b,e) -> e - b + 1) <$>
  getBounds (inputNodes ns)

structureOutputs :: NNStructure d -> Word
structureOutputs ns = unsafePerformIO $ (\(b,e) -> e - b + 1) <$>
  getBounds (outputNodes ns)

-- | A set of weight values to be used with an 'NNStructure'
data WeightValues = WeightValues {
  countWeightValues :: Word,
  weightValues :: ForeignPtr Double
 }

-- | The result of running a feed forward pass.
data FeedForward (d :: Bool) = FeedForward {
  ffBaseStructure :: NNStructure d,
  ffBaseWeights :: WeightValues,
  ffNodeGradients :: ForeignPtr Double,
  ffNodeOutputs :: ForeignPtr Double
 }

-- | A set of weight deltas
data WeightUpdate = WeightUpdate {
  weightUpdateCount :: Word,
  weightUpdate :: ForeignPtr Double
 }

-- | Represents the error gradient which has reached the input nodes during
-- backpropagation.
data InputTension = InputTension {
  tensionInputCount :: Word,
  inputTension :: ForeignPtr Double
 }

instance Functor (NNBuilder d s) where
  fmap f (NNBuilder s) = NNBuilder (\n w i o p -> let
    (n', w', i', o', p', a) = s n w i o p
    in (n', w', i', o', p', f a)
   )

instance Applicative (NNBuilder d s) where
  pure a = NNBuilder (\n w i o p -> (n, w, i, o, p, a))
  NNBuilder f <*> NNBuilder b = NNBuilder (\n0 w0 i0 o0 p0 -> let
    (n1, w1, i1, o1, p1, f') = f n0 w0 i0 o0 p0
    (n2, w2, i2, o2, p2, b') = b n1 w1 i1 o1 p1
    in (n2, w2, i2, o2, p2, f' b')
   )

instance Monad (NNBuilder d s) where
  return = pure
  NNBuilder a >>= f = NNBuilder (\n0 w0 i0 o0 p0 -> let
    (n1, w1, i1, o1, p1, a') = a n0 w0 i0 o0 p0
    NNBuilder b = f a'
    in b n1 w1 i1 o1 p1
   )

instance Show WeightValues where
  showsPrec _ wv = unsafePerformIO $ withForeignPtr (weightValues wv) $ \p ->
    let
      go :: Bool -> Word -> IO ShowS
      go d i
        | i == countWeightValues wv = touchForeignPtr (weightValues wv) >>
            return ('}':)
        | otherwise = do
          v <- peekElemOff p (fromIntegral i)
          r <- unsafeInterleaveIO $ go True (i + 1)
          let z = showsPrec 0 v . r
          return $ if d then (", "++) . z else z
      in (('{':) .) <$> go False 0

instance Read WeightValues where
  readsPrec p s = map (\(w,r) -> (packWeights w,r)) $
   readsPrec p $ map (\c -> case c of
     '{' -> '['
     '}' -> ']'
     _ -> c
    ) s

instance Show WeightUpdate where
  showsPrec _ wv = unsafePerformIO $ withForeignPtr (weightUpdate wv) $ \p ->
    let
      go :: Bool -> Word -> IO ShowS
      go d i
        | i == weightUpdateCount wv = touchForeignPtr (weightUpdate wv) >>
            return ('}':)
        | otherwise = do
          v <- peekElemOff p (fromIntegral i)
          r <- unsafeInterleaveIO $ go True (i + 1)
          let z = showsPrec 0 v . r
          return $ if d then (", "++) . z else z
      in (('{':) .) <$> go False 0

instance Semigroup WeightUpdate where
  (<>) = mappend

instance Monoid WeightUpdate where
  mempty = unsafePerformIO $ do
    p <- newForeignPtr_ nullPtr
    return $ WeightUpdate { weightUpdateCount = 0, weightUpdate = p }
  mappend a b = mconcat [a,b]
  mconcat [] = mempty
  mconcat [a] = a
  mconcat l = unsafePerformIO $ do
    let s = maximum $ map weightUpdateCount l
    f <- mallocForeignPtrArray (fromIntegral s)
    withForeignPtr f $ \p -> do
      forM_ [0 .. s - 1] $ \i -> pokeElemOff p (fromIntegral i) 0
      forM_ l $ \a -> withForeignPtr (weightUpdate a) $ \p' ->
        when (weightUpdateCount a /= 0) $
        forM_ [0 .. weightUpdateCount a - 1] $ \i -> do
          rt <- peekElemOff p (fromIntegral i)
          c <- peekElemOff p' (fromIntegral i)
          pokeElemOff p (fromIntegral i) (rt + c)
    return $ WeightUpdate { weightUpdateCount = s, weightUpdate = f }

-- | Adds more input nodes to the neural network and returns them
addInputs :: Word -> NNBuilder d s (Layer s)
addInputs d = NNBuilder (\n w i o p -> let
  n' = n + d
  e = n' - 1
  in (n', w, i <> mconcat (map pure [n .. e]), o, p, Layer (ILayer n e))
 )

layerSize :: Layer s -> Word
layerSize (Layer (ILayer n e)) = e - n + 1

addBaseWeights :: Word -> Word -> NNBuilder d s (WeightSelector s)
addBaseWeights piw pow = NNBuilder (\n w i o p -> let
  w' = piw * pow + w
  in (n, w', i, o, p, WS (IWeightSelector {
    weightsInputs = piw,
    weightsOutputs = pow,
    getWeight = \a ii oi -> a (w + ii + piw * oi),
    updateWeight = \a ii oi d -> a (w + ii + piw * oi) d
   }))
 )

fixedWeights :: Word -> Word -> (Word -> Word -> Double) -> WeightSelector s
fixedWeights piw pow d = WS (IWeightSelector {
  weightsInputs = piw,
  weightsOutputs = pow,
  getWeight = const $ \ii oi -> return (d ii oi),
  updateWeight = const $ const $ const $ const $ return ()
 })

linearLayer :: [(Layer s, WeightSelector s)] ->
  NNBuilder d s (Maybe (Layer s))
linearLayer [] = return Nothing
linearLayer l@((l1,w1):r) = let
  ls = wsOutputs w1
  in NNBuilder (\n w i o p -> let
    n' = n + ls
    e = n' - 1
    wo = mconcat <$> mapM (\(Layer (ILayer b e'), WS w0) ->
      if e' - b + 1 == weightsInputs w0 && weightsOutputs w0 == ls
        then Just $ pure $ WeightPatch b n w0
        else Nothing
     ) l
    in case wo of
      Nothing -> (n, w, i, o, p, Nothing)
      Just wo' -> (n + ls, w, i, o, p <> wo', Just (Layer (ILayer n e)))
 )

standardLayer :: [(Layer s, WeightSelector s)] -> ActivationFunction ->
  NNBuilder d s (Maybe (Layer s))
standardLayer [] _ = return Nothing
standardLayer l@((l1,w1):r) af = let
  ls = wsOutputs w1
  in NNBuilder (\n w i o p -> let
    n' = n + ls
    e = n' - 1
    wo = mconcat <$> mapM (\(Layer (ILayer b e'), WS w0) ->
      if e' - b + 1 == weightsInputs w0 && weightsOutputs w0 == ls
        then Just $ pure $ WeightPatch b n w0
        else Nothing
     ) l
    aaf = pure $ ApplyActivation n e af
    in case wo of
      Nothing -> (n, w, i, o, p, Nothing)
      Just wo' -> (n + ls, w, i, o, p <> wo' <> aaf, Just (Layer (ILayer n e)))
 )

pointwiseSum :: [Layer s] -> NNBuilder d s (Maybe (Layer s))
pointwiseSum = doPointwise PointwiseSum

pointwiseProduct :: [Layer s] -> NNBuilder d s (Maybe (Layer s))
pointwiseProduct = doPointwise PointwiseProduct

doPointwise :: (Word -> Word -> [(Word,Word)] -> NNOperation d) ->
  [Layer s] -> NNBuilder d s (Maybe (Layer s))
doPointwise _ [] = return Nothing
doPointwise cs ll = let
  ls = foldl1' lcm $ map layerSize ll
  in NNBuilder (\n w i o p -> let
    n' = n + ls
    e = n' - 1
    ap = pure $ cs n ls $ map (\l@(Layer (ILayer b _)) ->
      (b, layerSize l)
     ) ll
    in (n',w,i,o,p <> ap,Just (Layer (ILayer n e)))
   )

pointwiseUnary :: Layer s -> (Double -> (Double,Double)) ->
 NNBuilder d s (Layer s)
pointwiseUnary il@(Layer (ILayer b e)) f = NNBuilder (\n w i o p -> let
  l = layerSize il
  n' = n + l
  e' = n' - 1
  a = pure $ PointwiseUnary b n l f
  in (n',w,i,o,p <> a, Layer (ILayer n e'))
 )

stochasticLayer :: [(Layer s, WeightSelector s)] -> ActivationFunction ->
  (forall g . RandomGen g =>
    g -> Double -> Double -> (Double, Double,g)
   ) ->
  NNBuilder True s (Maybe (Layer s))
stochasticLayer ip af rf = NNBuilder (\n w i o p -> let
  NNBuilder sf = standardLayer ip af
  (n1,w1,i1,o1,p1,r) = sf n w i o p
  in case r of
    Nothing -> (n, w, i, o, p, Nothing)
    Just l@(Layer (ILayer b e)) -> (n1, w1, i1, o1, p1 <> pure (
      ApplyRandomization b e rf
     ), r)
 )

-- | Split a layer into two parts: the second parameter giving the size of the
-- first part. If the layer is too small: it will be returned directly and
-- the second part of the result will be 'Nothing'
splitLayer :: Layer s -> Word -> (Layer s, Maybe (Layer s))
splitLayer l@(Layer (ILayer b e)) s = if s >= e - b + 1
  then (l,Nothing)
  else let m = b + s in (Layer (ILayer b (m - 1)), Just $ Layer (ILayer m e))

-- | Split a layer into individual nodes.
layerNodes :: Layer s -> [Layer s]
layerNodes (Layer (ILayer b e)) = [Layer (ILayer x x) | x <- [b .. e]]

-- | Concatenate a list of layers into a single layer
concatLayers :: [Layer s] -> NNBuilder d s (Layer s)
concatLayers [] = fail "Tried to concatenate empty list of layers"
concatLayers l0 = let
  sc [] = undefined
  sc l@(Layer (ILayer b _):_) = go l where
    go [Layer (ILayer _ e)] = (Layer (ILayer b e),[])
    go (Layer (ILayer _ m):r@(Layer (ILayer n _):_))
      | m + 1 == n = go r
      | otherwise = (Layer (ILayer b m),r)
  cl z@(Layer (ILayer b _), _) = NNBuilder (\n w i o p -> (n,w,i,o,p,n)) >>=
    go z
   where
    go (Layer (ILayer b1 e1), r) n = do
      d <- NNBuilder (\n' w i o p -> let
        s = e1 - b1 + 1
        n2 = n' + s - 1
        in (n2,w,i,o,p <> pure (Copy b1 n' s),n2)
       )
      case r of
        [] -> return (Layer (ILayer n d))
        _ -> go (sc r) n
  in case sc l0 of
    (r,[]) -> return r
    z -> cl z

addOutputs :: Layer s -> NNBuilder d s ()
addOutputs (Layer (ILayer b e)) = NNBuilder (\n w i o p ->
  (n, w, i, o <> mconcat (map pure [b .. e]), p, ())
 )

runNNBuilder :: forall d a .
  (forall s . NNBuilder d s a) -> (NNStructure d, a)
runNNBuilder (NNBuilder bf) = let
  (n, w, i, o, p, a) = bf 1 0 mempty mempty mempty
  in unsafePerformIO $ do
    ia <- newArray (0, catTreeSize i - 1) 0
    oa <- newArray (0, catTreeSize o - 1) 0
    pa <- newArray (0, catTreeSize p - 1) undefined
    let
      fill :: MArray a e IO => CatTree e -> a Word e-> IO ()
      fill ct ar = foldr (\x r d -> do
        writeArray ar d x
        r (d + 1)
       ) (const $ return ()) ct 0
    fill i ia
    fill o oa
    fill p pa
    return (NNStructure {
      countNodes = n,
      countBaseWeights = w,
      inputNodes = ia,
      outputNodes = oa,
      nnOperations = pa
     },a)

initialWeights :: RandomGen g => NNStructure d -> g -> (Double,Double) ->
  (WeightValues, g)
initialWeights s g r = initialWeights' (countBaseWeights s) g r

initialWeights' :: RandomGen g => Word -> g -> (Double,Double) ->
  (WeightValues, g)
initialWeights' s g r = unsafePerformIO $ do
  f <- mallocForeignPtrArray (fromIntegral s)
  g' <- withForeignPtr f $ \p -> let
    go i g1
      | i == s = return g1
      | otherwise = let
        (v,g2) = randomR r g1
        in pokeElemOff p (fromIntegral i) v >> go (i + 1) g2
    in go 0 g
  return (WeightValues {weightValues = f, countWeightValues = s}, g')

serializeWeights :: WeightValues -> BS.ByteString
serializeWeights r = BS.fromForeignPtr
  (castForeignPtr $ weightValues r)
  0
  (fromIntegral $ countWeightValues r *
    fromIntegral (sizeOf (undefined :: Double)))
  

deserializeWeights :: BS.ByteString -> WeightValues
deserializeWeights s = let
  (bp, offset, len) = BS.toForeignPtr s
  in if offset == 0
    then WeightValues {
      weightValues = castForeignPtr bp,
      countWeightValues = fromIntegral $ len `div` 8
     }
    else unsafePerformIO $ do
      bp' <- mallocForeignPtrArray len
      withForeignPtr bp $ \p -> withForeignPtr bp' $ \p' ->
        forM_ [0 .. len - 1] $ \i -> do
          v <- peekElemOff p (i + offset)
          pokeElemOff p' i v
      return $ WeightValues {
        weightValues = castForeignPtr bp,
        countWeightValues = fromIntegral $ len `div` 8
       }

packWeights :: [Double] -> WeightValues
packWeights l = unsafePerformIO $ do
  let len = length l
  wv <- mallocForeignPtrArray $ len * sizeOf (undefined :: Double)
  withForeignPtr wv $ \p -> forM_ (zip [0 ..] l) $ \(i,v) ->
    pokeElemOff p i v
  return $ WeightValues {
    weightValues = wv,
    countWeightValues = fromIntegral len
   }

unpackWeights :: WeightValues -> [Double]
unpackWeights wv = unsafePerformIO $ withForeignPtr (weightValues wv) $ \p ->
 let
  go :: Word -> IO [Double]
  go n
    | fromIntegral n == countWeightValues wv = [] <$
      touchForeignPtr (weightValues wv)
    | otherwise = do
      v <- peekElemOff p (fromIntegral n)
      r <- unsafeInterleaveIO $ go (n + 1)
      return (v:r)
  in go 0

-- | The weighted sum of a list of weight updates (useful for implementing
-- momentum)
weightedSumUpdates :: [(Double,WeightUpdate)] -> WeightUpdate
weightedSumUpdates [] = mempty
weightedSumUpdates [(1,a)] = a
weightedSumUpdates l = unsafePerformIO $ do
    let s = maximum $ map (weightUpdateCount . snd) l
    f <- mallocForeignPtrArray (fromIntegral s)
    withForeignPtr f $ \p -> do
      forM_ [0 .. s - 1] $ \i -> pokeElemOff p (fromIntegral i) 0
      forM_ l $ \(m,a) -> withForeignPtr (weightUpdate a) $ \p' ->
        when (weightUpdateCount a /= 0) $
        forM_ [0 .. weightUpdateCount a - 1] $ \i -> do
          rt <- peekElemOff p (fromIntegral i)
          c <- peekElemOff p' (fromIntegral i)
          pokeElemOff p (fromIntegral i) (rt + c * m)
    return $ WeightUpdate { weightUpdateCount = s, weightUpdate = f }

feedForward ::
  NNStructure False -> WeightValues-> [Double] -> FeedForward False
stochasticFeedForward :: RandomGen g =>
  NNStructure d -> WeightValues -> [Double] -> g -> (FeedForward d, g)
(feedForward, stochasticFeedForward) = let
  init w s = do
    let a = mallocForeignPtrArray $ fromIntegral $ countNodes s
    o <- a
    g <- a
    forM_ [o,g] $ \f -> withForeignPtr f $ \p -> do
      pokeElemOff p 0 1
      forM_ [1 .. countNodes s - 1] $ \i -> pokeElemOff p (fromIntegral i) 0
    return $ FeedForward {
      ffBaseStructure = s,
      ffBaseWeights = w,
      ffNodeGradients = g,
      ffNodeOutputs = o
     }
  loadInputs f d = withForeignPtr (ffNodeOutputs f) $ \p -> do
    ab <- getBounds $ inputNodes $ ffBaseStructure f
    forM_ (zip (range ab) d) $ \(i,x) -> do
      n <- readArray (inputNodes $ ffBaseStructure f) i
      pokeElemOff p (fromIntegral n) x
  step :: Ptr Double -> Ptr Double -> Ptr Double -> NNOperation d -> IO ()
  step o g w p = case p of
    WeightPatch s t ws -> forM_ [0 .. weightsOutputs ws - 1] $ \j -> do
      v <- (sum <$>) $ forM [0 .. weightsInputs ws - 1] $ \i -> do
        ia <- peekElemOff o (fromIntegral (i + s))
        sw <- getWeight ws (peekElemOff w . fromIntegral) i j
        return (ia * sw)
      let i = fromIntegral (j + t)
      ov <- peekElemOff o i
      pokeElemOff o i (v + ov)
    ApplyActivation b e af -> do
      t <- forM [b .. e] $ \i -> peekElemOff o (fromIntegral i)
      forM_ (zip [b .. e] (activationFunction af t)) $ \(i, (a,g')) -> do
        pokeElemOff o (fromIntegral i) a
        pokeElemOff g (fromIntegral i) g'
    Copy s t l -> forM_ [0 .. l - 1] $ \i -> forM_ [o,g] $ \r ->
      peekElemOff r (fromIntegral (s + i)) >>=
        pokeElemOff r (fromIntegral (t + i))
    PointwiseSum t l a -> forM_ [0 .. l - 1] $ \i -> do
      v <- foldr (\(s,al) nxt v' -> do
        h <- peekElemOff o (fromIntegral s + fromIntegral (i `mod` al))
        let v2 = v' + h
        v2 `seq` nxt v2
       ) return a 0
      pokeElemOff g (fromIntegral (t + i)) 1
      pokeElemOff o (fromIntegral (t + i)) v
    PointwiseProduct t l a -> forM_ [0 .. l - 1] $ \i -> do
      v <- foldr (\(s,al) nxt v' -> do
        h <- peekElemOff o (fromIntegral s + fromIntegral (i `mod` al))
        let v2 = v' * h
        v2 `seq` nxt v2
       ) return a 1
      pokeElemOff o (fromIntegral (t + i)) v
      pokeElemOff g (fromIntegral (t + i)) v
    PointwiseUnary s t l f -> forM_ [0 .. l - 1] $ \i -> do
      x <- peekElemOff o (fromIntegral (s + i))
      let (y,y') = f x
      pokeElemOff o (fromIntegral (t + i)) y
      pokeElemOff g (fromIntegral (t + i)) y'
  ff s w i = unsafePerformIO $ do
    r <- init w s
    loadInputs r i
    (p0,pn) <- getBounds (nnOperations s)
    withForeignPtr (weightValues w) $ \w' ->
      withForeignPtr (ffNodeOutputs r) $ \o ->
        withForeignPtr (ffNodeGradients r) $ \g ->
          forM_ [p0 .. pn] $ \i -> do
            op <- readArray (nnOperations s) i
            step o g w' op
    return r
  in (ff, undefined)

getOutput :: FeedForward d -> Word -> Double
getOutput r i = unsafePerformIO $ do
  b <- getBounds $ outputNodes $ ffBaseStructure r
  if inRange b i
    then do
      ri <- readArray (outputNodes $ ffBaseStructure r) i
      v <- withForeignPtr (ffNodeOutputs r) $ \p ->
        peekElemOff p (fromIntegral ri)
      touchForeignPtr (ffNodeOutputs r)
      return v
    else
      return 0

getOutputs :: FeedForward d -> [Double]
getOutputs r = unsafePerformIO $ withForeignPtr (ffNodeOutputs r) $ \p -> do
  b <- getBounds $ outputNodes $ ffBaseStructure r
  let
    go [] = [] <$ touchForeignPtr (ffNodeOutputs r)
    go (i:n) = do
      ri <- readArray (outputNodes $ ffBaseStructure r) i
      v <- peekElemOff p (fromIntegral ri)
      c <- unsafeInterleaveIO (go n)
      return (v:c)
  go (range b)

backPropagate :: FeedForward d -> [Double] -> (WeightUpdate,InputTension)
backPropagate r e = unsafePerformIO $ do
  ne <- callocBytes
    (sizeOf (undefined :: Double) *
    (fromIntegral $ countNodes $ ffBaseStructure r))
  ob <- getBounds $ outputNodes $ ffBaseStructure r
  forM_ (zip (range ob) e) $ \(i,ev) -> do
    ni <- readArray (outputNodes $ ffBaseStructure r) i
    pokeElemOff ne (fromIntegral ni) ev
  (ai0,ain) <- getBounds $ nnOperations $ ffBaseStructure r
  wd <- mallocForeignPtrArray $ fromIntegral $
    countBaseWeights $ ffBaseStructure r
  withForeignPtr wd $ \wdp ->
    forM_ [0 .. countBaseWeights (ffBaseStructure r)] $ \i ->
      pokeElemOff wdp (fromIntegral i) 0
  withForeignPtr wd $ \wdp ->
    withForeignPtr (weightValues $ ffBaseWeights r) $ \w0 ->
    withForeignPtr (ffNodeGradients r) $ \g ->
    withForeignPtr (ffNodeOutputs r) $ \o ->
    forM_ [ain, ain - 1 .. ai0] $ \ai -> do
      a <- readArray (nnOperations $ ffBaseStructure r) ai
      case a of
        ApplyActivation b e af -> case backpropFunction af of
          Just bf -> do
            e' <- forM [b .. e] $ peekElemOff ne . fromIntegral
            g' <- forM [b .. e] $ peekElemOff g . fromIntegral
            forM_ (zip [b .. e] $ bf e' g') $ \(i,e2) ->
              pokeElemOff ne (fromIntegral i) e2
          Nothing -> forM_ [b .. e] $ \i' -> do
            let i = fromIntegral i'
            e' <- peekElemOff ne i
            g' <- peekElemOff g i
            pokeElemOff ne i (e' * g')
        WeightPatch s t ws -> forM_ [0 .. weightsOutputs ws - 1] $ \j -> do
          let j' = j + t
          e <- peekElemOff ne (fromIntegral j')
          forM_ [0 .. weightsInputs ws - 1] $ \i -> do
            let i' = i + s
            iv <- peekElemOff o (fromIntegral i')
            updateWeight ws (\ix v -> do
              v0 <- peekElemOff wdp (fromIntegral ix)
              pokeElemOff wdp (fromIntegral ix) (v + v0)) i j (iv * e)
            wij <- getWeight ws (peekElemOff w0 . fromIntegral) i j
            ie0 <- peekElemOff ne (fromIntegral i')
            pokeElemOff ne (fromIntegral i') (ie0 + wij * e)
        Copy s t l -> forM_ [0 .. l - 1] $ \i ->
          peekElemOff ne (fromIntegral (t + i)) >>=
            pokeElemOff ne (fromIntegral (s + i))
        PointwiseSum t l a -> forM_ a $ \(s,al) ->
          forM_ [0 .. al - 1] $ \i -> do
            v <- sum <$> forM [i, i + al .. l]
              (peekElemOff ne . fromIntegral . (+ t))
            ec <- peekElemOff ne (fromIntegral (i + s))
            pokeElemOff ne (fromIntegral (i + s)) (v + ec)
        PointwiseProduct t l a -> forM_ a $ \(s,al) -> do
          forM_ [0 .. al - 1] $ \i -> do
            f <- peekElemOff o (fromIntegral (s + i))
            (v,p) <- foldr (\i2 nxt (v',p') -> do
              v1 <- peekElemOff ne (fromIntegral (i2 + t))
              p1 <- peekElemOff o (fromIntegral (i2 + t))
              let v2 = v' + v1; p2 = p' + p1
              v2 `seq` p2 `seq` nxt (v2,p2)
             ) return [i, i + al .. l] (0,0)
            ec <- peekElemOff ne (fromIntegral (s + i))
            pokeElemOff ne (fromIntegral (s + i)) $ v * p / f + ec
        PointwiseUnary s t l f -> forM_ [0 .. l - 1] $ \i -> do
          v <- peekElemOff ne (fromIntegral (t + i))
          y' <- peekElemOff g (fromIntegral (t + i))
          x <- peekElemOff ne (fromIntegral (s + i))
          pokeElemOff ne (fromIntegral (s + i)) (x + v * y')
        _ -> return ()
  itb <- getBounds $ inputNodes $ ffBaseStructure r
  let itc = (\(a, b) -> max 0 (b - a + 1)) itb
  it <- mallocForeignPtrArray $ fromIntegral itc
  withForeignPtr it $ \t ->
    forM_ (range itb) $ \i -> do
      i' <- readArray (inputNodes $ ffBaseStructure r) i
      v <- peekElemOff ne $ fromIntegral i'
      pokeElemOff t (fromIntegral i) v
  free ne
  return (WeightUpdate {
    weightUpdateCount = countBaseWeights $ ffBaseStructure r,
    weightUpdate = wd
   }, InputTension {
     tensionInputCount = itc,
     inputTension = it
    })

inputError :: InputTension -> [Double]
inputError t = unsafePerformIO $ withForeignPtr (inputTension t) $ \p -> let
  go n
    | n == tensionInputCount t = [] <$ touchForeignPtr (inputTension t)
    | otherwise = do
      v <- peekElemOff p (fromIntegral n)
      r <- unsafeInterleaveIO $ go (n + 1)
      return (v:r)
  in go 0

applyDelta :: Double -> WeightValues -> WeightUpdate -> WeightValues
applyDelta lr = applyDeltaWith (\a b -> a + lr * b)

-- | Apply a weight delta using a helper function. The first argument to the
-- helper function is the current weight; the second is the corresponding weight
-- update value
applyDeltaWith :: (Double -> Double -> Double) -> 
  WeightValues -> WeightUpdate -> WeightValues
applyDeltaWith uf wv wu = unsafePerformIO $ do
  let s = max (countWeightValues wv) (weightUpdateCount wu)
  r <- mallocForeignPtrArray (fromIntegral s)
  withForeignPtr r $ \p -> withForeignPtr (weightValues wv) $ \p0 ->
    withForeignPtr (weightUpdate wu) $ \pd -> forM_ [0 .. s - 1] $ \i -> do
      a <- if i < countWeightValues wv
        then peekElemOff p0 (fromIntegral i)
        else return 0
      b <- if i < weightUpdateCount wu
        then peekElemOff pd (fromIntegral i)
        else return 0
      pokeElemOff p (fromIntegral i) (uf a b)
  return $ WeightValues {
    countWeightValues = s,
    weightValues = r
   }

-- Quick and dirty tree list. Won't bother balancing because we only need
-- to build and traverse: no need to lookup by index.
data CatTree a =
  Run !Word a |
  CatNode !Word (CatTree a) (CatTree a) |
  CatNil

catTreeSize (Run s _) = s
catTreeSize (CatNode s _ _) = s
catTreeSize CatNil = 0

instance (Eq a) => Eq (CatTree a) where
  a == b = catTreeSize a == catTreeSize b && case a of
    Run _ v -> foldr (\v' r -> v' == v && r) True b
    CatNode _ l r -> let
      (bl,br) = splitCT b (catTreeSize l)
      in l == bl && r == br
    CatNil -> foldr (const $ const False) True b

-- Current implementation breaks Eq instance
instance (Show a) => Show (CatTree a) where
  showsPrec _ t = ('[':) . go t . (']':) where
    go CatNil = id
    go (Run 1 v) = showsPrec 0 v
    go (Run n v) = showsPrec 8 v . (" * "++) . showsPrec 8 n
    go (CatNode _ l r) = go l . (", "++) . go r

instance Semigroup (CatTree a) where
  CatNil <> b = b
  a <> CatNil = a
  a <> b = CatNode (catTreeSize a + catTreeSize b) a b

instance Monoid (CatTree a) where
  mappend = (<>)
  mempty = CatNil

instance Functor CatTree where
  fmap f = go where
    go (Run l a) = Run l (f a)
    go (CatNode s a b) = CatNode s (go a) (go b)
    go CatNil = CatNil

instance Foldable CatTree where
  foldMap f = go where
    go (Run l a) = mt [(t,1),(mempty,0)] where
      t = f a
      mt (p1@(a1,s1):r1@(~((a2,s2):r))) = let
        sd = s1 + s1
        st = s1 + s2
        in case s1 `compare` l of
          GT -> mt r1 -- should be unreachable
          EQ -> a1
          LT -> case sd `compare` l of
            GT -> case st `compare` l of
              GT -> mt (p1:r)
              EQ -> mappend a1 a2
              LT -> mt ((mappend a1 a2, st):r)
            EQ -> mappend a1 a1
            LT -> mt ((mappend a1 a1, sd):p1:r1)
    go (CatNode _ a b) = mappend (go a) (go b)
    go CatNil = mempty

instance Traversable CatTree where
  sequenceA CatNil = pure CatNil
  sequenceA (CatNode s a b) = CatNode s <$> sequenceA a <*> sequenceA b
  sequenceA r@(Run 1 a) = Run 1 <$> a
  sequenceA r@(Run _ _) = sequenceA $ expandCT r

expandCT :: CatTree a -> CatTree a
expandCT (Run s0 a) = let
  e 0 = CatNil
  e 1 = Run 1 a
  e n = let
    n' = n `div` 2
    in CatNode n (e n') (e (n - n'))
  in e s0
expandCT (CatNode s a b) = CatNode s (expandCT a) (expandCT b)
expandCT CatNil = CatNil

splitCT :: CatTree a -> Word -> (CatTree a, CatTree a)
splitCT CatNil _ = (CatNil, CatNil)
splitCT r 0 = (CatNil, r)
splitCT r@(Run n a) s = if n <= s
  then (r,CatNil)
  else (Run s a, Run (n - s) a)
splitCT r@(CatNode n a b) s = if n <= s
  then (r,CatNil)
  else case catTreeSize a `compare` s of
    EQ -> (a, b)
    LT -> let
      (l,b') = splitCT b (s - catTreeSize a)
      in (a <> l, b')
    GT -> let
      (a',l) = splitCT a s
      in (a', l <> b)

instance Applicative CatTree where
  pure = Run 1
  CatNil <*> _ = CatNil
  _ <*> CatNil = CatNil
  r@(Run lf f) <*> a' = let
    rt = fmap f a'
    go (CatNode _ a b) = go a <> go b
    go (Run 1 _) = rt
    go r@(Run _ _) = go (expandCT r)
    go CatNil = CatNil
    in go r
  (CatNode _ a b) <*> r = (a <*> r) <> (b <*> r)

instance Monad CatTree where
  return = pure
  fail = const CatNil
  r@(Run _ a) >>= f = let
    rt = f a
    go (CatNode _ a b) = go a <> go b
    go (Run 1 _) = rt
    go r@(Run _ _) = go (expandCT r)
    go CatNil = CatNil
    in go r
  (CatNode _ a b) >>= f = (a >>= f) <> (b >>= f)
  CatNil >>= _ = CatNil

reverseCT :: CatTree a -> CatTree a
reverseCT r@(Run _ _) = r
reverseCT (CatNode l a b) = CatNode l (reverseCT b) (reverseCT a)
