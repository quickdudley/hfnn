module AI.HFNN.Bayesian (
 ) where

import Data.List (replicate)
import AI.HFNN.Internal
import System.Random

data WeightDistribution = WeightDistribution WeightValues WeightValues

newtype Deviation = Deviation WeightValues

prior :: Word -> WeightDistribution
prior n = let
  f = packWeights . replicate (fromIntegral n)
  in WeightDistribution (f 0) (f 1)

sample :: WeightDistribution -> Deviation -> WeightValues
sample (WeightDistribution m sd) (Deviation d') = let
  d = applyDeltaWith (*) sd (asUpdate d')
  in applyDeltaWith (+) m (asUpdate d)

updateDistribution ::
  Double -> WeightDistribution -> Deviation -> WeightUpdate ->
  WeightDistribution
updateDistribution a (WeightDistribution m sd) (Deviation d') u = let
  dsd = asUpdate $ applyDeltaWith (*) d' u
  in WeightDistribution
    (applyDelta a m u)
    (applyDelta a sd dsd)

deviation :: RandomGen g => Word -> g -> (Deviation, g)
deviation n' rng0 = let
  go 0 rng = ([],rng)
  go 1 rng = let
    (a,_,rng') = normalPair rng
    in ([a],rng')
  go n rng = let
    (a,b,rng1) = normalPair rng
    (r,rng2) = go (n - 2) rng1
    in (a:b:r, rng2)
  (dl, rngz) = go n' rng0
  in (Deviation (packWeights dl), rngz)

normalPair :: RandomGen g => g -> (Double,Double,g)
normalPair = go where
  go rng0 = let
    (v1,rng1) = randomR (-1,1) rng0
    (v2,rng2) = randomR (-1,1) rng1
    s = v1*v1 + v2*v2
    in if s >= 1
      then go rng2
      else if s == 0
        then (0,0,rng2)
        else let
          scale = sqrt (-2 * log s / s)
          in (v1 * scale, v2 * scale, rng2)
