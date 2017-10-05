{-# LANGUAGE RankNTypes,KindSignatures,DataKinds,GADTs #-}
module AI.HFNN.Internal (
  WeightSelector,
  Layer,
  NNBuilder,
  bias
 ) where

import Data.Semigroup
import Data.Word
import Foreign.Ptr
import Data.Array.Storable
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
  SoftMax :: Word -> Word -> Word -> NNOperation a
  PointwiseSum :: Word -> [Word] -> NNOperation a
  PointwiseProduct :: Word -> [Word] -> NNOperation a
  PointwiseUnary :: Word -> Word -> (Double -> (Double, Double)) ->
    NNOperation a

-- | A monad for assembling feedforward networks. The boolean type parameter
-- indicates whether or not the network may use stochastic units
newtype NNBuilder (d :: Bool) s a = NNBuilder (
  Word -> Word ->
  CatTree Word ->
  CatTree Word ->
  CatTree (NNOperation d) ->
  (Word, Word, CatTree Word, CatTree Word, CatTree (NNOperation d), a)
 )

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

addInputs :: Word -> NNBuilder d s (Layer s)
addInputs d = NNBuilder (\n w i o p -> let
  n' = n + d
  e = n' - 1
  in (n', w, i <> mconcat (map pure [n .. e]), o, p, Layer (ILayer n e))
 )

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

fixedWeights :: Word -> Word -> Double -> WeightSelector s
fixedWeights piw pow d = WS (IWeightSelector {
  weightsInputs = piw,
  weightsOutputs = pow,
  getWeight = const $ const $ const $ return d,
  updateWeight = const $ const $ const $ const $ return ()
 })

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
    go (Run l a) = mr l where
      t = f a
      mr 1 = t
      mr n = let
        (nl,m) = n `divMod` 2
        nr = nl + m
        in mappend (mr nl) (mr nr)
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
