#JC （JS CUDA）
##What is JC？
JC is a NodeJs module for doing linear algebra (which mainly consists of vector and matrix operations) based on CUDA.

It means you can easily use Javascript to bring your GPU down computing linear problems in parallel. In my experience, JC is **5 - 1000 times faster** than traditional CPU method thanks to huge development of modern GPU, and JC doesn't merely binding the CUDA ToolKit for using, it hides the relatively uncomprehensive BLAS routine or solvers' routine(in development) under the hood.

**And most important one is**: User can quickly deploy JC with very little effort instead of learning enormous and complicate API provided by CUDA Toolkit which is writen in C++.

##What JC can do?
 * Level-1
   1. arbitrary dimension vector add
   2. arbitrary dimension vector dot product
   3. arbitrary dimension vector Euclidean Norm
   4. arbitrary dimension vector multiply scalar
   
 * Level-2
   1. arbitrary dimension vector tensor product
   2. arbitrary dimension matrix add
   3. arbitrary dimension matrix multiply scalar
   4. arbitrary dimension matrix multiply vector
  
 * Level-3
   1. arbitrary dimension matrix multiply matrix
   2. arbitrary dimension matrix multiply matrix ( Batched )
 
 * Solvers
   1. quick dense n x n matrix inverse, n is less than 32 ( Batched )
   2. dense n x n matrix LU decomposition & solve linear system & matrix inverse ( Batched )
   3. dense n x n matrix QR decomposition & solve linear system & matrix inverse ( Batched ) *Coming soon*
   4. Sparse matrix solvers *Coming soon*

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

##Unit test
If you want to do unit tests, or take them as examples, please goto "./Pre-build" folder and copy whole "unit_test" folder into your module directory which contains ["JC.js", "jc.node", "JSCUDA.dll"].  
Then open the terminal in module's directory and input
```
E:\expamles>npm install
```
to install dependencies:
 1. mocha ^2.5.3 ( JavaScript test framework )
 2. colors ^1.1.2 ( Color and style outputs in node.js console )
 ( It's totally OK to install them globally as npm package by `npm install mocha -g` and `npm install colors -g` )

And now you can run test:
```
E:\expamles>mocha ${testname}.js
```
Output is something like this:
![test result][https://github.com/CallmeNezha/JSCUDA/blob/master/results/test.JPG]

##API Reference & Guide
[https://github.com/CallmeNezha/JSCUDA/wiki/]
