{-# LANGUAGE DataKinds #-}
import Control.Applicative
import Control.Monad
import System.Exit
import AI.HFNN
import System.Random

main :: IO ()
main = do
  g <- getStdGen
  let
    (w0,g') = initialWeights xorStructure g (-0.05,0.05)
    (w1,g1) = initialWeights minStructure g' (-0.05,0.05)
  setStdGen g1
  simpletest w1
  let wn = train 1000000 w0
  putStrLn ""
  putStrLn "trained xor output"
  forM_ samples $ \(i,_) ->
    print $ getOutputs $ feedForward xorStructure wn i

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
  Just h <- standardLayer [(i,h0w),(bias,h0b)] logistic
  ow <- addBaseWeights 4 1
  ob <- addBaseWeights 1 1
  Just o <- standardLayer [(h,ow),(bias,ob)] logistic
  addOutputs o

minStructure :: NNStructure False
minStructure = fst $ runNNBuilder $ do
  i <- addInputs 1
  w <- addBaseWeights 1 1
  wb <- addBaseWeights 1 1
  Just o <- standardLayer [(i,w),(bias,wb)] logistic
  addOutputs o
