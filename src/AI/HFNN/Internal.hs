{-# LANGUAGE RankNTypes #-}
module AI.HFNN.Internal (
  NetworkBuilder,
  WeightSelector,
  NetworkStructure
 ) where

import Foreign.ForeignPtr
import Foreign.Storable
import System.IO.Unsafe

data WeightSelector = WeightSelector (forall m . Monad m =>
  Int -> Int -> Double -> m Double
 ) Int Int

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

data NodeRange = NodeRange Int Int
data WeightRange = WeightRange Int Int

getInputNodes = NetworkBuilder (\s -> let
  n = structureInputs s
  in (s,NodeRange 1 n)
 )

getOutputNodes = NetworkBuilder (\s -> let
  i = structureInputs s
  o = structureOutputs s
  in (s,NodeRange (i + 1) (i + o))
 )

bias = NodeRange 0 0

newHiddenNodes n = NetworkBuilder (\s -> let
  x = structureTotalNodes s
  x' = x + n
  s' = s {structureTotalNodes = x'}
  in (s', NodeRange (x + 1) x')
 )
