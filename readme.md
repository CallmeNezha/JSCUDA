#JC （JS CUDA）
##What is JC？
JC is a NodeJs module for doing linear algebra (which mainly consists of vector and matrix operations) based on CUDA.

It means you can easily use Javascript to bring your GPU down computing linear problems. In my experience, JC is **5 - 1000 times faster** than traditional CPU method thanks to huge development of modern GPU, and JC doesn't merely binding the CUDA ToolKit for using, it hides the relatively uncomprehensive BLAS routine or solvers' routine(in development) under the hood.

**And most important one is**: User can quickly deploy JC with very little effort instead of learning enormous and complicate API provided by CUDA Toolkit which is writen in C++.

##Requirements
- [x] Make sure your graphic card support CUDA v7.5 and compute capability is higher than 2.0, if you are not sure please refer to [https://developer.nvidia.com/cuda-gpus]
- [x] Have Nodejs v6.2.X installed

##How to install?
###Pre-build
copy files in git ./Pre-build to any directory you like for example E:/Example/
 1. **jc.node**  // NodeJs module
 2. **JSCUDA.dll** // C++ binding part
 3. **JC.js** // Javascript part

###Build from source

##Usage
After copied related files to your module directory, all you have to do is require 'JC.js' as regular Node module.

Javascript:
```javascript
var JC = require( ${path_of_JC.js} )
```
Coffeescript:
```javascript
JC = require( ${path_of_JC.js} )
```
**Attension:** If you ignore the extension '.js', Node will require 'jc.node' instead, and 'jc.node' is Node's C++ binding part which offer basic data structures as well as basic linear functions supporting 'JC.js'. If you only want to use JC, please add '.js' explicitly.

Then you can use it like this

Coffeescript:
```coffeescript
#@@@ You can understand Host as CPU part, Device as GPU part)

JC = require( ${path_of_JC.js} )
testLength = 100
JC.cudaDeviceInit() # Preface

vAh = new Float32Array( ( Math.random() for num in [0...len] ) ) # Host memory
vAd = new JC.VectorD( vAh.length, vAh ) # Assign vAh value to Device memory
vBd = new JC.VectorD( vAh.length, vAh ) # Assgin same value to vBd

vAd.add( vBd ) # vAd = vAd + vBd

vAd.copyTo( vAd.length, vAh ) # Copy value to Host from Device memory

# Although Device memory will be freed automatically,
# it is recommanded to 'destroy' it explicitly,
# give back device memory to GPU for better performance
vAd.destroy()
console.log( #{ vAh } ) # output the result
JC.cudaDeviceReset()    # Before program exits

```

##API Reference & Guide
[https://github.com/CallmeNezha/JSCUDA/wiki/]
