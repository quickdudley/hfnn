{-# LANGUAGE DataKinds #-}
module AI.HFNN.Convolutional (

 ) where

import AI.HFNN
import Control.Monad
import Control.Applicative
import Data.Array
import Data.Word

-- | Create a convolutional layer.
convolutionalLayer :: Array (Word,Word) (Layer s) -- ^ Grid of 'Layer's
  -> (Word,Word) -- ^ Width and height of each receptive field
  -> (Word,Word) -- ^ Horizontal and vertical stride length
  -> (Word,Word) -- ^ Width and height of each output block
  -> (Word,Word) -- ^ Width and height of supplementary border
  -> Word -- ^ The number of channels in the output
  -> ActivationFunction -- ^ the activation function to use
  -> NNBuilder d s (Array (Word,Word) (Layer s))
convolutionalLayer ia (fw,fh) (sw,sh) (cw,ch) (bw,bh) c af = do
  let
    ((ix0,iy0),(ixn,iyn)) = bounds ia
    [iw,ih] = zipWith (\z n -> n - z + 1) [ix0,iy0] [ixn,iyn]
    ow = (iw - fw + 1) `div` sw * cw + 2 * bw
    oh = (ih - fh + 1) `div` sh * ch + 2 * bw
    bwb = ((0,0,0,0),(bw - 1, bh - 1, cw - 1, ch - 1))
    cwb = ((0,0,0,0),(fw - 1, fh - 1, cw - 1, ch - 1))
    cbb = ((0,0),(cw - 1, ch - 1))
    ic = layerSize $ ia ! (ix0,iy0)
  borderWeights <- (array bwb <$>) $ forM (range bwb) $ \z -> ((,) z) <$>
    addBaseWeights 1 c
  convBias <- (array cbb <$>) $ forM (range cbb) $ \z -> ((,) z) <$>
    addBaseWeights 1 c
  convWeights <- (array cwb <$>) $ forM (range cwb) $ \z -> ((,) z) <$>
    addBaseWeights ic c
  undefined