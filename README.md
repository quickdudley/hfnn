# Haskell Flexible Neural Networks

Another Haskell neural network and backpropagation library. It was initially
intended as a replacement for the neural network module in Varroa (which only
supported fully connected networks), and offers the following improvements.

 * Arbitrary feedforward neural network structures (recurrent neural networks
  and backpropagation through time will also be possible but will require some
  external plumbing)

 * Shared weight schemes (including but not limited to convolutional neural
  networks)

 * Straightforward and fast binary serialization

## HFNN's model

In several machine learning papers: we see neural networks described as
parameterized functions; denoted as fθ(x); where f is the structure of the
neural network; θ is the learnable parameters; and x is a given input.

### NNStructure

This type represents the "f" component in the notation above. Values of this
type can be created by calling `runNNBuilder` with an `NNBuilder` object
(described below). The boolean type parameter indicates whether or not the
structure contains stochastic units (not yet implemented)

### WeightValues

This type represents the "θ" component in the notation above. An initial value
can be created with the `initialWeights` or `initialWeights'` function; and
updated using the `applyDelta` function.

### WeightUpdate

This represents the partial derivatives of "θ" with respect to the loss
function (which is not always explicitly calculated: see below). Values of this
type can be created using the `backPropagate` function; and can be summed using
`mappend` or `<>`

### NNBuilder

This monad is used as a DSL for creating `NNStructure` objects. The basic
operations are `addInputs` which adds a layer of input nodes; `addBaseWeights`
which adds an abstract weight matrix (for which the actual values come from a
`WeightValues` object at feedforward time); `standardLayer` which adds a hidden
layer by applying abstract weight matrices to existing layers and applying an
activation function; and `addOutputs` which changes a hidden layer into an
output layer. There are convenience functions for creating layers without
having to care about the abstract matrices; but dealing with them is required
when implementing new weight sharing schemes.

### ActivationFunction

An object representing an activation function and the required information to
perform feedforward and backpropagation passes on layers using it. At pesent
this means the inner function must return the first derivative as well as the
immediate return value; but in some future version this approach may be replaced
with symbolic differentiation.

Some of the provided values of this type have odd names in order to prevent
collisions with prelude functions.

### FeedForward

Created by the `feedForward` or (currently unimplemented)
`stochatsticFeedForward` function. Activation levels of the output nodes can
be extracted with the `getOutput` and `getOutputs` functions; and this type is
one of the arguments to the `backPropagate` function.

### Back propagation

The arguments to the `backPropagate` function are: the `FeedForward` value; and
the negated partial derivaives of the loss function with respect to each output.
The `backpropExample` helper function is a wrapper for `backPropagate` that
minimises the sum of squared deviations (for which the partial derivatives are
simply the differences between the actual outputs and the targets).

## Wish List

There are some features which I want to include but will initially defer.

 * Offloading to GPU via OpenGL compute shaders

 