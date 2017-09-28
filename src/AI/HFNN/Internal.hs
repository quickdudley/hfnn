{-# LANGUAGE RankNTypes #-}
module AI.HFNN.Internal (

 ) where

import Data.Semigroup
import Foreign.Ptr
import Data.Array.Storable
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

-- Quick and dirty tree list. Won't bother balancing because we only need
-- to build and traverse: no need to lookup by index.
data CatTree a = Run !Int a | CatNode !Int (CatTree a) (CatTree a)

catTreeSize (Run s _) = s
catTreeSize (CatNode s _ _) = s

instance Semigroup (CatTree a) where
  a <> b = CatNode (catTreeSize a + catTreeSize b) a b

instance Functor CatTree where
  fmap f = go where
    go (Run l a) = Run l (f a)
    go (CatNode s a b) = CatNode s (go a) (go b)

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

-- Cannot implement "instance Traversable CatTree" without sacrificing
-- run-length compression.

instance Applicative CatTree where
  pure = Run 1
  (Run lf f) <*> a' = go a' where
    go (Run lr a) = Run (lf * lr) (f a)
    go (CatNode _ a b) = go a <> go b
  (CatNode _ a b) <*> r = (a <*> r) <> (b <*> r)

instance Monad CatTree where
  return = pure
  (Run l a) >>= f = go (f a) where
    go (Run l' b) = Run (l' * l) b
    go (CatNode _ x y) = go x <> go y
