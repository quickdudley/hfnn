{-# LANGUAGE RankNTypes #-}
module AI.HFNN.Internal (

 ) where

import Foreign.Ptr
import Foreign.ForeignPtr
import Foreign.Storable
import System.IO.Unsafe

import AI.HFNN.Activation

-- | Represents the relationship between a linear array of doubles and a
-- particular weight matrix
data WeightSelector = WeightSelector {
  weightsInputs :: Int,
  weightsOutputs :: Int,
  getWeight :: Ptr Double -> Int -> Int -> IO Double,
  updateWeight :: Ptr Double -> Int -> Int -> (Double -> Double) -> IO ()
 }