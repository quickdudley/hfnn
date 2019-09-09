{-# LANGUAGE DataKinds #-}
import Control.Applicative
import Control.Monad
import System.Exit
import AI.HFNN
import AI.HFNN.Bayesian
import System.Random

main :: IO ()
main = do
  g <- getStdGen
  let
    (w0,g') = initialWeights xorStructure g (-0.05,0.05)
    (w2,g1) = initialWeights softmaxStructure g' (-0.05,0.05)
    (twd,g2) = trainBayesian 1000000 g1 (prior (structureBaseWeights xorStructure))
    (sd,g3) = deviation (structureBaseWeights xorStructure) g2
    sr = sample twd sd
  setStdGen g3
  let wn = train 1000000 w0
  putStrLn ""
  putStrLn "trained xor output"
  forM_ samples $ \(i,_) ->
    print $ getOutputs $ feedForward xorStructure wn i
  let ws = trainSoftmax 1000000 w2
  putStrLn "untrained softmax output"
  forM_ [[if i == j then 1 else 0 | j <- [1,2,3]] | i <- [1,2,3]] $ \s ->
    print $ getOutputs $ feedForward softmaxStructure w2 s
  putStrLn "trained softmax output"
  forM_ [[if i == j then 1 else 0 | j <- [1,2,3]] | i <- [1,2,3]] $ \s ->
    print $ getOutputs $ feedForward softmaxStructure ws s
  putStrLn "trained xor output (bayes by backprop)"
  forM_ samples $ \(i,_) ->
    print $ getOutputs $ feedForward xorStructure sr i

simpletest :: WeightValues -> IO Bool
simpletest w = do
  let
    ff = feedForward minStructure w [1]
    (wu,_) = backPropagate ff $ zipWith (flip subtract) [1] $ getOutputs ff
    ff2 = feedForward minStructure (applyDelta 0.2 w wu) [1]
  putStrLn $ "Initial output: " ++ show (getOutputs ff)
  putStrLn $ "Increased output: " ++ show (getOutputs ff2)
  return $ foldr (&&) True $ zipWith (<) (getOutputs ff) (getOutputs ff2)

train :: Int -> WeightValues -> WeightValues
train 0 wv = wv
train n wv = let
  wu = mconcat $ map (\(i,o) -> singleSample i o wv) samples
  wv' = applyDelta 0.3 wv wu
  in wv' `seq` train (n - 1) wv'

trainBayesian :: RandomGen g => Int -> g -> WeightDistribution -> (WeightDistribution, g)
trainBayesian 0 rng wd = (wd, rng)
trainBayesian n rng wd = let
  (us, rng2) = foldr (\(i,o) rst rng' -> let
    (d,rng1) = deviation (structureBaseWeights xorStructure) rng'
    (r,rngr) = rst rng1
    w = sample wd d
    swu = singleSample i o w
    u = distributionUpdate d swu
    in (u:r, rngr)
   ) (\rng' -> ([],rng')) samples rng
  u = mconcat us
  wd' = updateDistribution 0.3 wd u
  in wd' `seq` trainBayesian (n - 1) rng2 wd'

singleSample :: [Double] -> [Double] -> WeightValues -> WeightUpdate
singleSample i o w = let
  ff = feedForward xorStructure w i
  (wu,_) = backpropExample ff (map Just o)
  in wu

samples :: [([Double],[Double])]
samples = [
  ([0,0],[0]),
  ([0,1],[1]),
  ([1,0],[1]),
  ([1,1],[0])
 ]

xorStructure :: NNStructure False
xorStructure = fst $ runNNBuilder $ do
  i <- addInputs 2
  h0w <- addBaseWeights 2 4
  h0b <- addBaseWeights 1 4
  ~(Just h) <- standardLayer [(i,h0w),(bias,h0b)] logistic
  ow <- addBaseWeights 4 1
  ob <- addBaseWeights 1 1
  ~(Just o) <- standardLayer [(h,ow),(bias,ob)] logistic
  addOutputs o

minStructure :: NNStructure False
minStructure = fst $ runNNBuilder $ do
  i <- addInputs 1
  w <- addBaseWeights 1 1
  wb <- addBaseWeights 1 1
  ~(Just o) <- standardLayer [(i,w),(bias,wb)] logistic
  addOutputs o

softmaxStructure :: NNStructure False
softmaxStructure = fst $ runNNBuilder $ do
  i <- addInputs 3
  w1 <- addBaseWeights 3 9
  b1 <- addBaseWeights 1 9
  w2 <- addBaseWeights 9 3
  b2 <- addBaseWeights 1 3
  ~(Just h) <- standardLayer [(i,w1),(bias,b1)] ahsin
  ~(Just o) <- standardLayer [(h,w2),(bias,b2)] softmax
  addOutputs o

trainSoftmax :: Int -> WeightValues -> WeightValues
trainSoftmax 0 wv = wv
trainSoftmax n wv = let
  c = map (\i -> map (\j -> if i == j then 1 else 0) [1,2,3]) [1,2,3]
  wu = mconcat $ map (\s -> let
   ff = feedForward softmaxStructure wv s
   (u,_) = backpropExample ff $ map Just s
   in u
   ) c
  wv' = applyDelta 0.8 wv wu
  in wv' `seq` trainSoftmax (n - 1) wv'
