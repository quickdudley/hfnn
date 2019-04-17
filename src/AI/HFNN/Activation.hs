{-# LANGUAGE RankNTypes #-}
module AI.HFNN.Activation (
  ActivationFunction(..),
  modifiedCuberoot,
  gaussian,
  logistic,
  htan,
  ahsin,
  relu,
  leakyRelu,
  sine,
  softplus,
  softmax
 ) where

import qualified Data.ByteString as BS

data ActivationFunction = ActivationFunction {
  activationFunction :: [Double] -> [(Double, Double)],
  glActivationFunction :: Maybe BS.ByteString,
  backpropFunction :: Maybe ([Double] -> [Double] -> [Double]),
  glBackpropFunction :: Maybe BS.ByteString
 }

nogl :: ActivationFunction
nogl = let
  errortext = "No GLSL template used directly"
  in ActivationFunction {
    activationFunction = error errortext,
    glActivationFunction = Nothing,
    backpropFunction = Nothing,
    glBackpropFunction = Nothing
   }

-- | Experimental activation function based on cube root. It should be less
-- prone to the vanishing gradient problem than the logistic function and other
-- sigmoid functions, and unlike cube root itself the derivative is never 1÷0.
modifiedCuberoot :: ActivationFunction
modifiedCuberoot = nogl {
  activationFunction = map $ \x' -> let
    x = abs x'
    crt = x ** (1/3)
    sr3 = 3 ** (1/2)
    in (signum x' * (
      crt - (atan (sr3 * crt))/sr3
     ), 1 / (3 * x**(2/3) + 1))
 }

logistic :: ActivationFunction
logistic = nogl {
  activationFunction = map $ \x -> let
    y = 1 / (1 + exp (-x))
    in (y, y * (1 - y))
 }

gaussian :: ActivationFunction
gaussian = nogl {
  activationFunction = map $ \x -> let
    y = exp(-(x^2))
    in (y,-2 * y * x)
 }

-- | Hyperbolic tangent
htan :: ActivationFunction
htan = nogl {
  activationFunction = map $ \x -> let t = tanh x in (t, 1 - t^2)
 }

-- | Inverse hyperbolic sine
ahsin :: ActivationFunction
ahsin = nogl {
  activationFunction = map $ \x -> (asinh x, 1 / sqrt (x ^ 2 + 1))
 }

relu :: ActivationFunction 
relu = nogl {
  activationFunction = map $ \x -> if x < 0
    then (0, 0)
    else (x, 1)
 }

leakyRelu :: Double -> ActivationFunction
leakyRelu r = nogl {
  activationFunction = map $ \x -> if x < 0
    then (x * r, r)
    else (x, 1)
 }

sine :: ActivationFunction
sine = nogl {
  activationFunction = map $ \x -> (sin x, cos x)
 }

softplus :: ActivationFunction
softplus = nogl {
  activationFunction = map $ \x -> (log (1 + exp x), 1 / (1 + exp (-x)))
 }

softmax :: ActivationFunction
softmax = nogl {
  activationFunction = \z -> let
    z_exp = map exp z
    z_exp_sum = sum z_exp
    in map (\x -> let y = x / z_exp_sum in (y,y)) z_exp,
  backpropFunction = Just $ \g e -> zipWith (\i g' ->
    sum (zipWith3 (\j g2 e' -> let
      k = if i == j then 1 else 0
      in (k - g2) * e'
     ) [0 ..] g e) * g'
   ) [0 ..] g
 }
