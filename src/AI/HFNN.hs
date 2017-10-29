{-# LANGUAGE DataKinds #-}
module AI.HFNN (
  module AI.HFNN.Internal,
  module AI.HFNN.Activation,
  simpleLayer,
  backpropExample
 ) where

import Control.Applicative
import Control.Monad
import AI.HFNN.Activation
import AI.HFNN.Internal

simpleLayer :: [Layer s] -> Word -> ActivationFunction ->
  NNBuilder d s (Layer s)
simpleLayer l' s af = do
  let l = bias : l
  w <- forM l $ \l1 -> addBaseWeights (layerSize l1) s
  Just nl <- standardLayer (zip l w) af
  return nl

backpropExample :: FeedForward d -> [Maybe Double] ->
  (WeightUpdate, InputTension)
backpropExample ff ex = let
  er = zipWith (\e o -> case e of
    Just e' -> e' - o
    _ -> 0
   ) ex (getOutputs ff)
  in backPropagate ff er
