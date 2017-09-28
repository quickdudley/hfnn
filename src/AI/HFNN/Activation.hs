{-# LANGUAGE RankNTypes #-}
module AI.HFNN.Activation (
  ActivationFunction(..),
  modifiedCuberoot,
  logistic,
  relu,
  softplus
 ) where

import qualified Data.ByteString as BS

data DerivativeBase = X | Y deriving (Ord,Eq)

data ActivationFunction = ActivationFunction {
  activationFunction :: forall x . (Ord x,Floating x) => x -> x,
  derivativeBase :: DerivativeBase,
  afDerivative :: forall x . (Ord x,Floating x) => x -> x,
  glslActivationFunction :: Maybe BS.ByteString,
  glslDerivative :: Maybe BS.ByteString
 }

nogl :: ActivationFunction
nogl = let
  errortext = "No GLSL template used directly"
  in ActivationFunction (error errortext) (error errortext) (error errortext)
    Nothing Nothing

-- | Experimental activation function based on cube root. It should be less
-- prone to the vanishing gradient problem than the logistic function and other
-- sigmoid functions, and unlike cube root itself the derivative is never 1/0.
modifiedCuberoot :: ActivationFunction
modifiedCuberoot = nogl {
  activationFunction = \x' -> let
    x = abs x'
    crt = x ** (1/3)
    sr3 = 3 ** (1/2)
    in signum x' * (
      crt - (atan (sr3 * crt))/sr3
     ),
  derivativeBase = X,
  afDerivative = \x' -> let
    x = abs x'
    in 1 / (3 * x**(2/3) + 1)
}

logistic :: ActivationFunction
logistic = nogl {
  activationFunction = \x -> 1 / (1 + exp (-x)),
  derivativeBase = Y,
  afDerivative = \y -> y * (1 - y)
}

relu :: ActivationFunction 
relu = nogl {
  activationFunction = \x -> max x 0,
  derivativeBase = Y,
  afDerivative = \y -> if y < 0 then 0 else 1
}

softplus :: ActivationFunction
softplus = nogl {
  activationFunction = \x -> log (1 + exp x),
  derivativeBase = X,
  afDerivative = activationFunction logistic
}
