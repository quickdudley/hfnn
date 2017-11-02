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
    iab@((ix0,iy0),(ixn,iyn)) = bounds ia
    [iw,ih] = zipWith (\z n -> n - z + 1) [ix0,iy0] [ixn,iyn]
    ow = (iw - fw + 1) `div` sw * cw + 2 * bw
    oh = (ih - fh + 1) `div` sh * ch + 2 * bw
    bwb = ((0,0,0,0),(fw - 1, fh - 1, cw - 1, ch - 1))
    cwb = ((0,0,0,0),(fw - 1, fh - 1, cw - 1, ch - 1))
    cbb = ((0,0),(cw - 1, ch - 1))
    ob = ((0,0),(ow - 1, oh - 1))
    ic = layerSize $ ia ! (ix0,iy0)
  borderWeights <- (array bwb <$>) $ forM (range bwb) $ \z -> ((,) z) <$>
    addBaseWeights 1 c
  convBias <- (array cbb <$>) $ forM (range cbb) $ \z -> ((,) z) <$>
    addBaseWeights 1 c
  convWeights <- (array cwb <$>) $ forM (range cwb) $ \z -> ((,) z) <$>
    addBaseWeights ic c
  (array ob <$>) $ forM (range ob) $ \(x,y) -> do
    let
      [(cx,icx),(cy,icy)] = zipWith divMod [x,y] [cw,ch]
      cbw = convBias ! (icx,icy)
      i = (bias,cbw) : do
        (ofx,ofy) <- range ((0,0),(fw - 1, fh - 1))
        let
          ix = ix0 + cx * sw + ofx
          iy = iy0 + cy * sh + ofy
        if inRange iab (ix,iy)
          then return (ia ! (ix,iy), convWeights ! (ofx, ofy, icx, icy))
          else return (bias, borderWeights ! (ofx, ofy, icx, icy))
    pl <- standardLayer i af
    case pl of
      Just pl' -> return ((x,y),pl')
      _ -> error "Convolutional layer: input layer has inconsistent number \
        \of channels"
