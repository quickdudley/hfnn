module AI.HFNN.Bayesian (
  WeightDistribution,
  Deviation,
  prior,
  sample,
  updateDistribution,
  distributionUpdate,
  deviation
 ) where

import Data.List (replicate)
import AI.HFNN.Internal
import System.Random

data WeightDistribution = WeightDistribution !WeightValues !WeightValues
data DistributionUpdate = DistributionUpdate WeightUpdate WeightUpdate

newtype Deviation = Deviation WeightValues

prior :: Word -> WeightDistribution
prior n = let
  f = packWeights . replicate (fromIntegral n)
  in WeightDistribution (f 0) (f 1)

sample :: WeightDistribution -> Deviation -> WeightValues
sample (WeightDistribution m sd) (Deviation d') = let
  d = applyDeltaWith (*) sd (asUpdate d')
  in applyDeltaWith (+) m (asUpdate d)

distributionUpdate :: Deviation -> WeightUpdate -> DistributionUpdate
distributionUpdate (Deviation d) u =
  DistributionUpdate u (asUpdate $ applyDeltaWith (*) d u)

instance Semigroup DistributionUpdate where
  DistributionUpdate m1 sd1 <> DistributionUpdate m2 sd2 =
    DistributionUpdate (m1 <> m2) (sd1 <> sd2)

instance Monoid DistributionUpdate where
  mappend = (<>)
  mempty = DistributionUpdate mempty mempty
  mconcat l = DistributionUpdate
    (mconcat $ map (\(DistributionUpdate m _) -> m) l)
    (mconcat $ map (\(DistributionUpdate _ sd) -> sd) l)

updateDistribution ::
  Double -> WeightDistribution -> DistributionUpdate ->
  WeightDistribution
updateDistribution a (WeightDistribution m sd) (DistributionUpdate mu sdu) =
  WeightDistribution
    (applyDelta a m mu)
    (applyDeltaWith (\v u -> abs (v + a * u)) sd sdu)

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

instance Show WeightDistribution where
  showsPrec n (WeightDistribution m sd) = let
    b = if n == 0 then id else \p -> ('(':) . p . (')':)
    in b $ ("WeightDistribution { mean = "++) . showsPrec 9 m .
      (", standard deviation = "++) . showsPrec 9 sd . ('}':)
