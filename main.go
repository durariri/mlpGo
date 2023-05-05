package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"math/rand"
	"time"
)

func target(X *mat.Dense) *mat.Dense {
	x1 := X.ColView(0).(*mat.VecDense)
	x2 := X.ColView(1).(*mat.VecDense)
	x3 := X.ColView(2).(*mat.VecDense)

	// Calculate y1 (first column in target matrix Y with func values)
	y1 := make([]float64, X.RawMatrix().Rows)
	for i := 0; i < len(y1); i++ {
		y1[i] = math.Log(math.Exp(math.Cos(x1.AtVec(i)))) + math.Tan(x2.AtVec(i)) + 1/math.Tan(x3.AtVec(i))
	}

	// Calculate y2 (second column in target matrix Y with {0; 1})
	y1Mean := mat.Sum(mat.NewVecDense(len(y1), y1)) / float64(len(y1))
	y2 := make([]float64, len(y1))
	for i, y1i := range y1 {
		if y1i > y1Mean {
			y2[i] = 1
		} else {
			y2[i] = 0
		}
	}

	// Combine y1 and y2 horizontally
	Y := mat.NewDense(len(y1), 2, nil)
	Y.SetCol(0, y1)
	Y.SetCol(1, y2)

	return Y
}

func main() {
	net := CreateNetwork(3, 3, 2, 0.01)
	rand.Seed(time.Now().UnixNano())

	// Create first row of arguments
	firstRow := []float64{6, 7, 8}
	rowCount := 24
	colCount := len(firstRow)

	// Create data slice for matrix
	data := make([]float64, rowCount*colCount)

	index := 0
	for i := 0; i < len(firstRow); i++ {
		for j := 0; j < len(firstRow); j++ {
			for k := 0; k < len(firstRow); k++ {
				if index >= 72 {
					break
				}
				data[index] = firstRow[i]
				data[index+1] = firstRow[j]
				data[index+2] = firstRow[k]
				index += 3
			}
		}
	}

	// Create matrix
	X := mat.NewDense(rowCount, colCount, data)
	Y := target(X)
	fmt.Printf("%v\n", mat.Formatted(Y))

	fmt.Printf("%v\n", mat.Formatted(X))
	var (
		Ymax = 1.0
		Ymin = 0.0
		//Xmax float64 = 8
		//Xmin float64 = 6
	)
	//for i := 0; i < 24; i++ {
	//	for j := 0; j < 3; j++ {
	//		X.Set(i, j, (X.At(i, j)-Xmin)/(Xmax-Xmin))
	//	}
	//}
	// Find Ymin and Ymax
	for i := 0; i < 24; i++ {
		if Y.At(i, 0) > Ymax {
			Ymax = Y.At(i, 0)
		}
		if Y.At(i, 0) < Ymin {
			Ymin = Y.At(i, 0)
		}
	}
	// Normalize Y
	for i := 0; i < 24; i++ {
		Y.Set(i, 0, (Y.At(i, 0)-Ymin)/(Ymax-Ymin))
	}
	fmt.Printf("\n%v\n%v\n", mat.Formatted(X), mat.Formatted(Y))
	fmt.Printf("\nFormatted \n%v\n\n%v\n%v\n", mat.Formatted(X.Slice(4, 24, 0, 3)), mat.Formatted(Y.Slice(4, 24, 0, 2)), mat.Formatted(X.Slice(0, 4, 0, 3)))
	Training(&net, mat.DenseCopyOf(X.Slice(4, 24, 0, 3)), mat.DenseCopyOf(Y.Slice(4, 24, 0, 2)))
	Predicting(&net, mat.DenseCopyOf(X.Slice(0, 3, 0, 3)))
	fmt.Printf("\nCompare\n%v\n", mat.Formatted(Y.Slice(0, 3, 0, 2)))
}

func Training(net *Network, X *mat.Dense, Y *mat.Dense) {
	var setSum = mat.NewDense(10000, 20, nil)
	for epochs := 0; epochs < 10000; epochs++ {
		for sets := 0; sets < 20; sets++ {
			inputs := make([]float64, net.inputs)
			for i := range inputs {
				inputs[i] = X.At(sets, i)
			}
			targets := make([]float64, net.outputs)
			for i := range targets {
				targets[i] = Y.At(sets, i)
			}
			setSum.Set(epochs, sets, net.Train(inputs, targets))
		}
	}
	for i := 0; i < setSum.RawMatrix().Rows; i += 1000 {
		row := setSum.RowView(i)
		rowSum := (mat.Sum(row)) / 20.0
		fmt.Printf("sum of the outpurErrors at %v epoch is %v\n", i, rowSum)
	}
}

func Predicting(net *Network, X *mat.Dense) {
	var comp = mat.NewDense(3, 2, nil)
	for sets := 0; sets < 3; sets++ {
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			inputs[i] = X.At(sets, i)
		}
		comp.SetRow(sets, net.Predict(inputs))
	}
	fmt.Printf("\nCompare\n%v\n", mat.Formatted(comp))
}

// Network is a neural network with 3 layers
type Network struct {
	inputs        int        // the number of neurons in the input layer
	hiddens       int        // the number of neurons in the hidden layer
	outputs       int        // the number of neurons in the output layer
	hiddenWeights *mat.Dense // matrix that represent the weights from the input to hidden layers
	outputWeights *mat.Dense // matrix that represent the weights from the hidden to output layers
	learningRate  float64    // the learning rate for the network
}

// CreateNetwork creates a neural network with random weights
// the hidden and output weights are randomly created
func CreateNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))
	return
}

// Train the neural network
func (net *Network) Train(inputData []float64, targetData []float64) float64 {
	// feedforward
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := subtract(targets, finalOutputs)
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	sumError := (outputErrors.At(0, 0)*outputErrors.At(0, 0) + outputErrors.At(1, 0)*outputErrors.At(1, 0)) / 2

	// backpropagate
	net.outputWeights = add(net.outputWeights,
		scale(net.learningRate,
			dot(multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = add(net.hiddenWeights,
		scale(net.learningRate,
			dot(multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)

	return sumError
}

func (net *Network) Predict(inputData []float64) []float64 {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	temp := []float64{
		finalOutputs.At(0, 0),
		finalOutputs.At(1, 0),
	}
	return temp
}

func sigmoid(_, _ int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

//
// Helper functions to allow easier use of Gonum
//

//product of two matrices
func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

//allows us to apply a function to the matrix
func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

// allows us to scale a matrix i.e. multiply a matrix by a scalar
func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

// multiplies 2 functions together (this is different from dot product`
func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

// allow to add or subtract a function to/from another
func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

// randomly generate a float64 array
func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}
	data = make([]float64, size)
	for i := 0; i < size; i++ {
		// data[i] = rand.NormFloat64() * math.Pow(v, -0.5)
		data[i] = dist.Rand()
	}
	return
}
