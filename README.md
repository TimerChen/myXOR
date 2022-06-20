# myXOR
A naive implementation of RAID4 Erasure Code.

## test

Ref: https://blog.csdn.net/oqqYuan1234567890/article/details/107702117

```shell
cd examples
go build simple-encode.go
go build simple-decode.go
./simple-encoder ../LICENSE
rm ../LICENSE.1
./simple-decoder -out LICENSE ../LICENSE
```