// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/popcode"
	"github.com/emer/etable/etensor"
)

// ExEnv is an example environment, that sets a single input point in a 2D
// input state and two output states as the X and Y coordinates of point.
// It can be used as a starting point for writing your own Env, without
// having much existing code to rewrite.
type ExEnv struct {
	Nm      string `desc:"name of this environment"`
	Dsc     string `desc:"description of this environment"`
	Size    int    `desc:"size of each dimension in 2D input"`
	MinDist float32
	MaxDist int
	//MaxAngle    int
	//WineUnits   int
	Wine1input etensor.Float32 // adding the first wine. Is it a etensor.Float32?
	Wine2input etensor.Float32 // added
	SweetDry   etensor.Float32
	LightFull  etensor.Float32
	AttnDim    etensor.Float32 //determine if you care about sweet/dry or light/full
	//Wine1Pop    popcode.TwoD    // hidden layer for wine 1 -> reflavored AlloInput
	//Wine2Pop    popcode.TwoD    // hidden layer for wine 2 -> reflavored EgoInput
	//CombinedPop popcode.TwoD    // large hidden layer that gets input from both other hidden layers
	SweetDryPop  popcode.OneD //is this something other than a 1x1 etensor?
	LightFullPop popcode.OneD //determining if something is in the light or full dimension
	DimSwitch    bool
	Winner       int
	//	W1Input     etensor.Float32 `desc: Hidden layer 1 input state, 2D Size x Size"`

	//	W2Input     etensor.Float32 `desc: Hidden layer 1 input state, 2D Size x Size"`
	//	Things I didn't think we needed anymore?
	NSDDistUnits int
	//NAngleUnits int
	//DistPop      popcode.OneD `desc:"population encoding of distance value"`
	//AnglePop     popcode.Ring
	//AttnPop      popcode.TwoD `desc:"2D population encoding of attn"`
	//AlloInputPop popcode.TwoD
	//EgoInputPop  popcode.TwoD
	//Point  image.Point `desc:"X,Y coordinates of point"`
	//Point2 image.Point
	//Point3 image.Point
	//Attn         etensor.Float32 `desc: "attentional layer"`
	// X        etensor.Float32 `desc:"X as a one-hot state 1D Size"`
	// Y        etensor.Float32 `desc:"Y  as a one-hot state 1D Size"`
	//Distance etensor.Float32
	//Angle    etensor.Float32
	//DistVal  float32
	//AngVal   float32
	Run   env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch env.Ctr `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial env.Ctr `view:"inline" desc:"trial increments over input states -- could add Event as a lower level"`
}

func (ev *ExEnv) Name() string { return ev.Nm }
func (ev *ExEnv) Desc() string { return ev.Dsc }

// Config sets the size, number of trials to run per epoch, and configures the states
func (ev *ExEnv) Config(sz int, ntrls int) {
	ev.Size = sz
	//ev.Wine1Pop.Defaults()
	//ev.Wine1Pop.Min = mat32.NewVec2(-1, -1) //WHAT IS THE PURPOSE FOR THIS?
	//ev.Wine1Pop.Max = mat32.NewVec2(float32(sz+3), float32(sz+3))
	//ev.Wine1Pop.Sigma.Set(0.1, 0.1)
	//ev.Wine2Pop.Defaults()
	//ev.Wine2Pop.Min = mat32.NewVec2(-3, -3)
	//ev.Wine2Pop.Max = mat32.NewVec2(float32(sz+5), float32(sz+5))
	// ev.Wine2Pop.Sigma.Set(0.1, 0.1)
	// ev.CombinedPop.Defaults()
	// Things I didn't think were necessary anymore?
	//ev.MaxDist = int(float64(sz) * math.Sqrt(2))
	//ev.MinDist = 5
	//ev.MaxAngle = 360
	//ev.NAngleUnits = 24
	//ev.NDistUnits = 10
	//ev.DistPop.Defaults()
	//ev.DistPop.Min = float32(ev.MaxDist) * -0.1
	//ev.DistPop.Max = float32(ev.MaxDist) * 1.1
	//ev.AnglePop.Defaults()
	//ev.AnglePop.Min = 0
	//ev.AnglePop.Max = 360
	//ev.AlloInputPop.Defaults()
	//ev.EgoInputPop.Defaults()
	//ev.AttnPop.Defaults()
	//ev.AttnPop.Min = mat32.NewVec2(-1, -1)
	//ev.AttnPop.Max = mat32.NewVec2(float32(sz+3), float32(sz+3))
	//ev.AlloInputPop.Min = mat32.NewVec2(-3, -3)
	//ev.AlloInputPop.Max = mat32.NewVec2(float32(sz+5), float32(sz+5))
	//ev.AlloInputPop.Sigma.Set(0.1, 0.1)
	//ev.EgoInputPop.Min = mat32.NewVec2(0, 0)
	//ev.EgoInputPop.Max = mat32.NewVec2(float32(sz*2+5), float32(sz*2+5))
	//ev.EgoInputPop.Sigma.Set(0.1, 0.1)

	currentTime := time.Now()
	rand.Seed(int64(currentTime.Unix()))

	ev.Trial.Max = ntrls

	ev.NSDDistUnits = 10

	//setting the shape of the big layers- nope
	///ev.Wine1Pop.SetShape([]int{sz, sz}, nil, []string{"Y", "X"})
	//ev.Wine2Pop.SetShape([]int{sz, sz}, nil, []string{"Y", "X"})

	//setting the shapes to a 1x
	ev.Wine1input.SetShape([]int{ev.Size}, nil, []string{"Wine1input"})
	ev.Wine2input.SetShape([]int{ev.Size}, nil, []string{"Wine2input"})
	//ev.LightFull.SetShape([]int{ev.NSDDistUnits}, nil, []string{"LightFull"})
	ev.SweetDry.SetShape([]int{ev.NSDDistUnits}, nil, []string{"SweetDry"})
	//ev.AttnDim.SetShape([]int{1}, nil, []string{"AttnDim"})

	//ev.LightFullPop.Defaults()
	//ev.LightFullPop.Min = float32(-1 * sz)
	//ev.LightFullPop.Max = float32(sz)
	//ev.LightFullPop.Sigma = 0.1
	ev.SweetDryPop.Defaults()
	ev.SweetDryPop.Min = float32(-1 * sz)
	ev.SweetDryPop.Max = float32(sz)
	ev.SweetDryPop.Sigma = 0.1

	//ev.EgoInput.SetShape([]int{sz*2 - 1, sz*2 - 1}, nil, []string{"Y", "X"})
	//ev.Attn.SetShape([]int{sz, sz}, nil, []string{"Y", "X"})
	//ev.AlloInput.SetShape([]int{sz + 3, sz + 3}, nil, []string{"Y", "X"})
	// ev.X.SetShape([]int{sz}, nil, []string{"X"})
	// ev.Y.SetShape([]int{sz}, nil, []string{"Y"})
	//ev.Distance.SetShape([]int{ev.NDistUnits}, nil, []string{"Distance"})
	//ev.Angle.SetShape([]int{ev.NAngleUnits}, nil, []string{"Angle"})
}

func (ev *ExEnv) Validate() error {
	if ev.Size == 0 {
		return fmt.Errorf("ExEnv: %v has size == 0 -- need to Config", ev.Nm)
	}
	return nil
}

func (ev *ExEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (ev *ExEnv) States() env.Elements {
	els := env.Elements{
		{"Wine1input", []int{ev.Size}, []string{"Wine1input"}},
		{"Wine2input", []int{ev.Size}, []string{"Wine2input"}},
		{"SweetDry", []int{ev.Size * 2}, []string{"SweetDry"}},
		//{"LightFull", []int{ev.Size * 2}, []string{"LightFull"}},
		//{"AttnDim", []int{1}, []string{"AttnDim"}},
	}
	return els
}

func (ev *ExEnv) State(element string) etensor.Tensor {
	switch element {
	case "Wine1input":
		return &ev.Wine1input
	case "Wine2input":
		return &ev.Wine2input
	case "SweetDry":
		return &ev.SweetDry
		//case "LightFull":
		//	return &ev.LightFull
		//case "AttnDim":
		//	return &ev.AttnDim
	}
	return nil
}

func (ev *ExEnv) Actions() env.Elements {
	return nil
}

// String returns the current state as a string DO YOU NEED THIS ANYMORE??
func (ev *ExEnv) String() string {
	return fmt.Sprintf("Pt") //, ev.Point.X, ev.Point.Y)
}

// Init is called to restart environment
func (ev *ExEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

// // NewPoint generates a new point and sets state accordingly
// func (ev *ExEnv) NewPoint() {
// 	//ev.Point.X = rand.Intn(ev.Size)
// 	//ev.Point.Y = rand.Intn(ev.Size)
// 	// ev.Point.X = 1
// 	// ev.Point.Y = 1
// 	/*for {
// 		ev.Point2.X = rand.Intn(ev.Size)
// 		ev.Point2.Y = rand.Intn(ev.Size)
// 		if ev.Point2 != ev.Point {
// 			break
// 		}
// 	}*/
// 	//maxDist1 := math.Hypot(float64(7-ev.Point.X), float64(7-ev.Point.Y)) // point 9, 9
// 	//maxDist2 := math.Hypot(float64(ev.Point.X), float64(ev.Point.Y))     // point 0, 0
// 	//ev.MaxDist = int(math.Min(maxDist1, maxDist2))
// 	ev.MinDist = 2
// 	ev.MaxDist = ev.Size - 1
// 	dist := ev.MinDist + rand.Float32()*(float32(ev.MaxDist)-ev.MinDist)
// 	ang := rand.Float32() * 360
// 	ev.Point3.X = 8 //ev.Size-1
// 	ev.Point3.Y = 8 //ev.Size-1
// 	for {
// 		ev.Point3.X = 8 + int(float64(dist*mat32.Cos(ang*math.Pi/180)))
// 		ev.Point3.Y = 8 + int(float64(dist*mat32.Sin(ang*math.Pi/180)))
// 		if !(ev.Point3.X == 8 && ev.Point3.Y == 8) { // point 3 cannot be 8, 8
// 			break
// 		}
// 	}
// 	xDist := ev.Point3.X - ev.Size
// 	yDist := ev.Point3.Y - ev.Size
// 	maxX := 0
// 	minX := 0
// 	maxY := 0
// 	minY := 0
// 	if xDist > 0 {
// 		maxX = ev.Size - xDist
// 		minX = 0
// 	}
// 	if xDist < 0 {
// 		maxX = ev.Size
// 		minX = int(math.Abs(float64(xDist)))
// 	}
// 	if xDist == 0 {
// 		minX = 0
// 		maxX = ev.Size
// 	}
// 	if yDist > 0 {
// 		maxY = ev.Size - yDist
// 		minY = 0
// 	}
// 	if yDist < 0 {
// 		maxY = ev.Size
// 		minY = int(math.Abs(float64(yDist)))
// 	}
// 	if yDist == 0 {
// 		minY = 0
// 		maxY = ev.Size
// 	}
// 	ev.Point.X = int(float32(minX) + rand.Float32()*float32((maxX-minX)))
// 	ev.Point.Y = int(float32(minY) + rand.Float32()*float32((maxY-minY)))
// 	ev.Point2.X = ev.Point.X + xDist
// 	ev.Point2.Y = ev.Point.Y + yDist
// 	//generate Point based on range above
// 	hypotDist := math.Hypot(float64(ev.Point2.X-ev.Point.X), float64(ev.Point2.Y-ev.Point.Y))
// 	xDistance := ev.Point2.X - ev.Point.X
// 	yDistance := ev.Point2.Y - ev.Point.Y

// 	ang0 := 0.0
// 	ang360 := 0.0
// 	if xDistance >= 0 && yDistance >= 0 {
// 		ang0 = math.Atan2(float64(yDistance), float64(xDistance)) * 180 / math.Pi
// 	} else if xDist < 0 && yDist >= 0 {
// 		ang0 = math.Atan2(float64(yDistance), float64(xDistance)) * 180 / math.Pi
// 	} else if xDist >= 0 && yDist < 0 {
// 		ang360 = 360 - (math.Abs(math.Atan2(float64(yDistance), float64(xDistance))) * 180 / math.Pi)
// 	} else { //xDist < 0 and yDist < 0
// 		ang360 = 360 + (math.Atan2(float64(yDistance), float64(xDistance)) * 180 / math.Pi)
// 	}
// 	ang = float32(ang0 + ang360)

// 	//ev.Point3.X = ev.Size - 1 + xDist
// 	//ev.Point3.Y = ev.Size - 1 + yDist

// 	ev.EgoInput.SetZeros()
// 	ev.Attn.SetZeros()
// 	ev.AlloInput.SetZeros()
// 	ev.Attn.SetFloat([]int{ev.Point.Y, ev.Point.X}, 1)
// 	//ev.AlloInput.SetFloat([]int{ev.Point.Y, ev.Point.X}, 1)
// 	//ev.AlloInput.SetFloat([]int{ev.Point2.Y, ev.Point2.X}, 1)
// 	//ev.EgoInput.SetFloat([]int{ev.Size - 1, ev.Size - 1}, 1) //center point of input
// 	//ev.EgoInput.SetFloat([]int{ev.Point3.Y, ev.Point3.X}, 1)
// 	ev.DistPop.Encode(&ev.Distance.Values, float32(hypotDist), ev.NDistUnits, false)
// 	ev.AnglePop.Encode(&ev.Angle.Values, float32(ang), ev.NAngleUnits)
// 	ev.AttnPop.Encode(&ev.Attn, mat32.NewVec2(float32(ev.Point.Y), float32(ev.Point.X)), false)
// 	//ev.EgoInputPop.Encode(&ev.EgoInput, mat32.NewVec2(float32(ev.Size-1), float32(ev.Size-1)), false)
// 	ev.EgoInputPop.Encode(&ev.EgoInput, mat32.NewVec2(float32(ev.Point3.Y), float32(ev.Point3.X)), true)
// 	ev.AlloInputPop.Encode(&ev.AlloInput, mat32.NewVec2(float32(ev.Point.Y), float32(ev.Point.X)), false)
// 	ev.AlloInputPop.Encode(&ev.AlloInput, mat32.NewVec2(float32(ev.Point2.Y), float32(ev.Point2.X)), true)
// 	ev.DistVal = float32(hypotDist)
// 	ev.AngVal = float32(ang)
// }

func (ev *ExEnv) NewCompare() {
	W1loc := 1
	W2loc := 1
	sd := 0
	ev.Wine1input.SetZeros()
	ev.Wine2input.SetZeros()

	for {
		W1loc = rand.Intn(ev.Size)
		W2loc = rand.Intn(ev.Size)

		//ev.AttnDim.SetZeros()

		//ev.Wine1input.SetFloat([]int{W1loc}, 1)
		//ev.Wine2input.SetFloat([]int{W2loc}, 1)
		//ev.AttnDim.SetFloat([]int{0}, 1)

		if W1loc != W2loc {
			break
		}
	}
	if W1loc > W2loc {
		ev.Winner = 1
	}
	if W1loc < W2loc {
		ev.Winner = 2
	}

	ev.Wine1input.SetFloat([]int{W1loc}, 1)
	ev.Wine2input.SetFloat([]int{W2loc}, 1)

	if ev.Winner == 1 {
		sd = -1 * ev.Size
	} else {
		sd = ev.Size
	}

	//ev.DimSwitch = erand.BoolProb(0.5, -1)
	ev.SweetDryPop.Encode(&ev.SweetDry.Values, float32(sd), len(ev.SweetDry.Values), false)
	//ev.LightFullPop.Encode(&ev.LightFull.Values, 0, 1, false)

}

// Step is called to advance the environment state
func (ev *ExEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.NewCompare()
	if ev.Trial.Incr() { // true if wraps around Max back to 0
		ev.Epoch.Incr()
	}
	return true
}

func (ev *ExEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *ExEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*ExEnv)(nil)
