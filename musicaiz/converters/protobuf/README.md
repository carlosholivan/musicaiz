
# Installing protobuf and `protoc` command
## Windows

1. Go to [protobuf releases](https://github.com/protocolbuffers/protobuf/releases/) and download the `...winXX.zip`
2. Add the path to the `proto.exe` to the env vars (or `set PATH=<paath_to_proto.exe>`)

## Linux


## Generate the .py files from .proto files

From the root path of this library run:
```
protoc musicaiz/converters/protobuf/musicaiz.proto --python_out=. --pyi_out=.
```

## Try it out

````
import musicaiz
midi = musicaiz.loaders.Musa("tests/fixtures/midis/mz_332_1.mid", structure="bars")
midi.to_proto()
````