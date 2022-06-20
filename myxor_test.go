package myxor

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

func TestEncoding(t *testing.T) {

	t.Run("default", func(t *testing.T) {
		testEncoding(t)
	})

	// t.Run(fmt.Sprintf("opt-%d", 0), func(t *testing.T) {
	// 	testEncoding(t)
	// })
}

// matrix sizes to test.
// note that par1 matric will fail on some combinations.
var testSizes = [][2]int{
	{1, 0}, {3, 0}, {5, 0}, {8, 0}, {10, 0}, {12, 0}, {14, 0}, {41, 0}, {49, 0},
	{1, 1},
	// {1, 2},
	{3, 1}, {5, 1}, {8, 1}, {10, 1}, {12, 1}, {14, 1}, {41, 1}, {49, 1}, {5, 1}}
var testDataSizes = []int{10, 100, 1000, 10001, 100003, 1000055}
var testDataSizesShort = []int{10, 10001, 100003}

func testEncoding(t *testing.T) {
	for _, size := range testSizes {
		data, parity := size[0], size[1]
		// fmt.Printf("test: %d %d\n", data, parity)
		rng := rand.New(rand.NewSource(0xabadc0cac01a))
		t.Run(fmt.Sprintf("%dx%d", data, parity), func(t *testing.T) {
			sz := testDataSizes
			if testing.Short() {
				sz = testDataSizesShort
			}
			for _, perShard := range sz {
				// fmt.Printf("[%d]\t", perShard)
				t.Run(fmt.Sprint(perShard), func(t *testing.T) {

					r, err := New(data, parity, WithAutoGoroutines(int(perShard)))
					if err != nil {
						t.Fatal(err)
					}
					shards := make([][]byte, data+parity)
					for s := range shards {
						shards[s] = make([]byte, perShard)
					}

					for s := 0; s < len(shards); s++ {
						rng.Read(shards[s])
					}

					err = r.Encode(shards)
					if err != nil {
						t.Fatal(err)
					}
					ok, err := r.Verify(shards)
					if err != nil {
						t.Fatal(err)
					}
					if !ok {
						t.Fatal("Verification failed")
					}

					if parity == 0 {
						// Check that Reconstruct and ReconstructData do nothing
						err = r.ReconstructData(shards)
						if err != nil {
							t.Fatal(err)
						}
						err = r.Reconstruct(shards)
						if err != nil {
							t.Fatal(err)
						}

						// Skip integrity checks
						return
					}

					// Delete one in data
					idx := rng.Intn(data)
					want := shards[idx]
					shards[idx] = nil

					err = r.ReconstructData(shards)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(shards[idx], want) {
						t.Fatal("did not ReconstructData correctly")
					}

					// Delete one randomly
					idx = rng.Intn(data + parity)
					want = shards[idx]
					shards[idx] = nil
					err = r.Reconstruct(shards)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(shards[idx], want) {
						t.Fatal("did not Reconstruct correctly")
					}

					err = r.Encode(make([][]byte, 1))
					if err != ErrTooFewShards {
						t.Errorf("expected %v, got %v", ErrTooFewShards, err)
					}

					// Make one too short.
					shards[idx] = shards[idx][:perShard-1]
					err = r.Encode(shards)
					if err != ErrShardSize {
						t.Errorf("expected %v, got %v", ErrShardSize, err)
					}
				})
			}
		})

	}
}
