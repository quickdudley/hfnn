# Haskell Flexible Neural Networks

Another Haskell neural network and backpropagation library. It is initially
intended as a replacement for the neural network module in Varroa, and will
offer the following improvements.

 * Arbitrary feedforward neural network structures (recurrent neural networks
  and backpropagation through time will also be possible but will require some
  external plumbing)

 * Shared weight schemes (including but not limited to convolutional neural
  networks)

 * Straightforward and fast binary serialization

## Wish List

There are some features which I want to include but will initially defer.

 * Offloading to GPU via OpenGL compute shaders

 