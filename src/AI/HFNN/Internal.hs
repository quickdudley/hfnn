{-# LANGUAGE RankNTypes #-}
module AI.HFNN.Internal (
  NetworkBuilder,
  WeightSelector,
  NetworkStructure,
  NodeRange,
 getInputNodes,
  getOutputNodes,
  bias,
  newHiddenNodes,
  newWeights
 ) where

import Foreign.ForeignPtr
import Foreign.Storable
import System.IO.Unsafe

-- | Represents the relationship between a linear array of doubles and a
-- particular weight matrix
data WeightSelector = WeightSelector (forall m . Monad m =>
  (Int -> Double -> m Double) ->
  Int -> Int -> Double -> m Double
 ) Int Int

-- | Describes the overall structure of a neural network, but not any of its
-- weights.
data NetworkStructure = NetworkStructure {
  structureInputs :: Int,
  structureOutputs :: Int,
  structureTotalNodes :: Int,
  structureWeights :: Int,
  structureFeedforward :: forall m . Monad m =>
    (WeightSelector -> Int -> Int -> m ()) -> m (),
  structureBackprop :: forall m . Monad m =>
    (WeightSelector -> Int -> Int -> Double -> m ()) -> m ()
 }

-- | Monad used for composing neural networks
newtype NetworkBuilder a = NetworkBuilder (
  NetworkStructure -> (NetworkStructure,a)
 )

instance Functor NetworkBuilder where
  fmap f (NetworkBuilder a) = NetworkBuilder (\s -> fmap f (a s))

instance Applicative NetworkBuilder where
  pure a = NetworkBuilder (\s -> (s,a))
  NetworkBuilder f <*> NetworkBuilder a = NetworkBuilder (\s0 -> let
    (s1,f') = f s0
    (s2,a') = a s1
    in (s2, f' a')
   )

instance Monad NetworkBuilder where
  return = pure
  NetworkBuilder a >>= f = NetworkBuilder (\s0 -> let
    (s1,a') = a s0
    NetworkBuilder b = f a'
    in b s1
   )

-- | A contiguous set of nodes
data NodeRange = NodeRange Int Int

-- | Gets the set of nodes used for input to the network
getInputNodes :: NetworkBuilder NodeRange
getInputNodes = NetworkBuilder (\s -> let
  n = structureInputs s
  in (s,NodeRange 1 n)
 )

-- | Gets the set of nodes used for output from the network
getOutputNodes :: NetworkBuilder NodeRange
getOutputNodes = NetworkBuilder (\s -> let
  i = structureInputs s
  o = structureOutputs s
  in (s,NodeRange (i + 1) (i + o))
 )

-- | The bias node
bias :: NodeRange
bias = NodeRange 0 0

-- | Returns a new layer of hidden nodes
newHiddenNodes :: Int -> NetworkBuilder NodeRange
newHiddenNodes n = NetworkBuilder (\s -> let
  x = structureTotalNodes s
  x' = x + n
  s' = s {structureTotalNodes = x'}
  in (s', NodeRange (x + 1) x')
 )

-- | Returns a new set of network weights not attached to any nodes
newWeights :: Int -> Int -> NetworkBuilder WeightSelector
newWeights i o = NetworkBuilder (\s -> let
  w = structureWeights s
  w' = i * o + w
  s' = s {structureWeights = w'}
  in (s', WeightSelector (\l x y v -> l (w + y + o * x) v) i o)
 )
