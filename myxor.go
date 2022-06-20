// Author: Jingxiao Chen
// Code Ref:
// 		1. https://github.com/klauspost/reedsolomon
// 		2. XORSIMD

// We implemented the whole EC algorithm follow klauspost/reedsolomon

package myxor

import (
	"errors"
	"io"
	"sync"
)

type Option func(*options)

type options struct {
	maxGoroutines int
	minSplitSize  int
	shardSize     int
	perRound      int

	// useAVX512, useAVX2, useSSSE3, useSSE2 bool
	// usePAR1Matrix                         bool
	// useCauchy                             bool
	fastOneParity bool
	// inversionCache                        bool

	// stream options
	// concReads  bool
	// concWrites bool
	// streamBS   int
}

type Encoder interface {
	Encode(shards [][]byte) error

	// EncodeIdx will add parity for a single data shard.
	// Parity shards should start out as 0. The caller must zero them.
	// Data shards must be delivered exactly once. There is no check for this.
	// The parity shards will always be updated and the data shards will remain the same.
	// EncodeIdx(dataShard []byte, idx int, parity [][]byte) error

	// Verify returns true if the parity shards contain correct data.
	// The data is the same format as Encode. No data is modified, so
	// you are allowed to read from data while this is running.
	// Verify(shards [][]byte) (bool, error)
	Reconstruct(shards [][]byte) error

	// ReconstructData will recreate any missing data shards, if possible.
	//
	// Given a list of shards, some of which contain data, fills in the
	// data shards that don't have data.
	//
	// The length of the array must be equal to Shards.
	// You indicate that a shard is missing by setting it to nil or zero-length.
	// If a shard is zero-length but has sufficient capacity, that memory will
	// be used, otherwise a new []byte will be allocated.
	//
	// If there are too few shards to reconstruct the missing
	// ones, ErrTooFewShards will be returned.
	//
	// As the reconstructed shard set may contain missing parity shards,
	// calling the Verify function is likely to fail.
	ReconstructData(shards [][]byte) error

	// Update parity is use for change a few data shards and update it's parity.
	// Input 'newDatashards' containing data shards changed.
	// Input 'shards' containing old data shards (if data shard not changed, it can be nil) and old parity shards.
	// new parity shards will in shards[DataShards:]
	// Update is very useful if  DataShards much larger than ParityShards and changed data shards is few. It will
	// faster than Encode and not need read all data shards to encode.
	// Update(shards [][]byte, newDatashards [][]byte) error

	// Split a data slice into the number of shards given to the encoder,
	// and create empty parity shards.
	//
	// The data will be split into equally sized shards.
	// If the data size isn't dividable by the number of shards,
	// the last shard will contain extra zeros.
	//
	// There must be at least 1 byte otherwise ErrShortData will be
	// returned.
	//
	// The data will not be copied, except for the last shard, so you
	// should not modify the data of the input slice afterwards.
	Split(data []byte) ([][]byte, error)

	Verify(shards [][]byte) (bool, error)

	Join(dst io.Writer, shards [][]byte, outSize int) error
}

const (
	avx2CodeGenMinSize       = 64
	avx2CodeGenMinShards     = 3
	avx2CodeGenMaxGoroutines = 8

	intSize = 32 << (^uint(0) >> 63) // 32 or 64
	maxInt  = 1<<(intSize-1) - 1
)

type myXor struct {
	DataShards   int // Number of data shards, should not be modified.
	ParityShards int // Number of parity shards, should not be modified.
	Shards       int // Total number of shards. Calculated, and should not be modified.
	// m            matrix
	// tree         *inversionTree
	parity [][]byte
	o      options
	mPool  sync.Pool
}

var defaultOptions = options{
	maxGoroutines: 384,
	minSplitSize:  -1,
	fastOneParity: false,
}

var ErrInvShardNum = errors.New("cannot create Encoder with less than one data shard or less than zero parity shards")
var ErrMaxShardNum = errors.New("cannot create Encoder with more than 256 data+parity shards")
var ErrMaxParityShards = errors.New("cannot create Encoder with more than 1 parity shards")

func New(dataShards, parityShards int, opts ...Option) (Encoder, error) {
	r := myXor{
		DataShards:   dataShards,
		ParityShards: parityShards,
		Shards:       dataShards + parityShards,
		o:            defaultOptions,
	}

	// for _, opt := range opts {
	// 	opt(&r.o)
	// }
	if dataShards <= 0 || parityShards < 0 {
		return nil, ErrInvShardNum
	}

	// Only support with GF(2)
	if parityShards > 1 {
		return nil, ErrMaxParityShards
	}

	if dataShards+parityShards > 256 {
		return nil, ErrMaxShardNum
	}

	if parityShards == 0 {
		return &r, nil
	}

	var err error = nil
	// switch {
	// case r.o.fastOneParity && parityShards == 1:
	// 	r.m, err = buildXorMatrix(dataShards, r.Shards)
	// case r.o.useCauchy:
	// 	r.m, err = buildMatrixCauchy(dataShards, r.Shards)
	// case r.o.usePAR1Matrix:
	// 	r.m, err = buildMatrixPAR1(dataShards, r.Shards)
	// default:xo
	// 	r.m, err = buildMatrix(dataShards, r.Shards)
	// }
	if err != nil {
		return nil, err
	}

	// Calculate what we want per round
	// r.o.perRound = cpuid.CPU.Cache.L2

	// divide := parityShards + 1
	return &r, err
}

func xorEncode(src [][]byte, dst []byte, n int) {
	// if len(dst) == nil {
	// 	return
	// }

	for i := 0; i < n; i++ {
		s := src[0][i]
		for j := 1; j < len(src); j++ {
			s ^= src[j][i]
		}
		dst[i] = s
	}
}

var ErrTooFewShards = errors.New("too few shards given")

// Encode parity for a set of data shards.
// An array 'shards' containing data shards followed by parity shards.
// The number of shards must match the number given to New.
// Each shard is a byte array, and they must all be the same size.
// The parity shards will always be overwritten and the data shards
// will remain the same.

func (r *myXor) Encode(shards [][]byte) error {

	if len(shards) != r.Shards {
		return ErrTooFewShards
	}

	err := checkShards(shards, false)
	if err != nil {
		return err
	}

	if r.ParityShards == 0 {
		return nil
	}

	// Get the slice of output buffers.
	output := shards[r.DataShards]
	// Do the coding.
	xorEncode(shards[0:r.DataShards], output, len(shards[0]))
	return nil
}

func (r *myXor) Reconstruct(shards [][]byte) error {
	return r.reconstruct(shards, false)
}

func (r *myXor) ReconstructData(shards [][]byte) error {
	return r.reconstruct(shards, true)
}

func (r *myXor) reconstruct(shards [][]byte, dataOnly bool) error {
	if len(shards) != r.Shards {
		return ErrTooFewShards
	}
	// Check arguments.
	err := checkShards(shards, true)
	if err != nil {
		return err
	}
	shardSize := shardSize(shards)

	numberPresent := 0
	dataPresent := 0
	for i := 0; i < r.Shards; i++ {
		if len(shards[i]) != 0 {
			numberPresent++
			if i < r.DataShards {
				dataPresent++
			}
		}
	}
	if numberPresent == r.Shards || dataOnly && dataPresent == r.DataShards {
		// Cool.  All of the shards data data.  We don't
		// need to do anything.
		return nil
	}

	// More complete sanity check
	if numberPresent < r.DataShards {
		return ErrTooFewShards
	}

	subShards := make([][]byte, r.DataShards)
	validIndices := make([]int, r.DataShards)
	invalidIndices := make([]int, 0)
	subMatrixRow := 0
	for matrixRow := 0; matrixRow < r.Shards && subMatrixRow < r.DataShards; matrixRow++ {
		if len(shards[matrixRow]) != 0 {
			subShards[subMatrixRow] = shards[matrixRow]
			validIndices[subMatrixRow] = matrixRow
			subMatrixRow++
		} else {
			invalidIndices = append(invalidIndices, matrixRow)
		}
	}

	if len(invalidIndices) == 0 && dataOnly {
		return nil
	}

	if len(invalidIndices) > 0 {
		shards[invalidIndices[0]] = make([]byte, shardSize)
		xorEncode(subShards, shards[invalidIndices[0]], shardSize)
	} else {
		if dataOnly {
			return nil
		}
		shards[r.DataShards] = make([]byte, shardSize)
		xorEncode(subShards, shards[r.DataShards], shardSize)
	}

	return nil
}

// Verify returns true if the parity shards contain the right data.
// The data is the same format as Encode. No data is modified.
func (r *myXor) Verify(shards [][]byte) (bool, error) {
	if len(shards) != r.Shards {
		return false, ErrTooFewShards
	}
	err := checkShards(shards, false)
	if err != nil {
		return false, err
	}

	if r.ParityShards == 0 {
		return true, nil
	}

	// Slice of buffers being checked.
	toCheck := shards[r.DataShards]

	// Do the checking.
	n := len(shards[0])

	for i := 0; i < n; i++ {
		s := shards[0][i]
		for j := 1; j < r.DataShards; j++ {
			s ^= shards[j][i]
		}
		if s != toCheck[i] {
			return false, nil
		}
	}

	return true, nil
}

var ErrReconstructRequired = errors.New("reconstruction required as one or more required data shards are nil")

// Join the shards and write the data segment to dst.
//
// Only the data shards are considered.
// You must supply the exact output size you want.
//
// If there are to few shards given, ErrTooFewShards will be returned.
// If the total data size is less than outSize, ErrShortData will be returned.
// If one or more required data shards are nil, ErrReconstructRequired will be returned.
func (r *myXor) Join(dst io.Writer, shards [][]byte, outSize int) error {
	// Do we have enough shards?
	if len(shards) < r.DataShards {
		return ErrTooFewShards
	}
	shards = shards[:r.DataShards]

	// Do we have enough data?
	size := 0
	for _, shard := range shards {
		if shard == nil {
			return ErrReconstructRequired
		}
		size += len(shard)

		// Do we have enough data already?
		if size >= outSize {
			break
		}
	}
	if size < outSize {
		return ErrShortData
	}

	// Copy data to dst
	write := outSize
	for _, shard := range shards {
		if write < len(shard) {
			_, err := dst.Write(shard[:write])
			return err
		}
		n, err := dst.Write(shard)
		if err != nil {
			return err
		}
		write -= n
	}
	return nil
}

// func encode(dst []byte, src [][]byte) {
// 	if supportsUnaligned {
// 		fastEncode(dst, src, len(dst))
// 	} else {
// 		// TODO(hanwen): if (dst, a, b) have common alignment
// 		// we could still try fastEncode. It is not clear
// 		// how often this happens, and it's only worth it if
// 		// the block encryption itself is hardware
// 		// accelerated.
// 		safeEncode(dst, src, len(dst))
// 	}

// }

// // fastEncode xor in bulk. It only works on architectures that
// // support unaligned read/writes.
// func fastEncode(dst []byte, src [][]byte, n int) {
// 	w := n / wordSize
// 	if w > 0 {
// 		wordBytes := w * wordSize

// 		wordAlignSrc := make([][]byte, len(src))
// 		for i := range src {
// 			wordAlignSrc[i] = src[i][:wordBytes]
// 		}
// 		fastEnc(dst[:wordBytes], wordAlignSrc)
// 	}

// 	for i := n - n%wordSize; i < n; i++ {
// 		s := src[0][i]
// 		for j := 1; j < len(src); j++ {
// 			s ^= src[j][i]
// 		}
// 		dst[i] = s
// 	}
// }

// func fastEnc(dst []byte, src [][]byte) {
// 	dw := *(*[]uintptr)(unsafe.Pointer(&dst))
// 	sw := make([][]uintptr, len(src))
// 	for i := range src {
// 		sw[i] = *(*[]uintptr)(unsafe.Pointer(&src[i]))
// 	}

// 	n := len(dst) / wordSize
// 	for i := 0; i < n; i++ {
// 		s := sw[0][i]
// 		for j := 1; j < len(sw); j++ {
// 			s ^= sw[j][i]
// 		}
// 		dw[i] = s
// 	}
// }
