JC = require("../linear")

date = new Date()
JC.cudaDeviceInit()

testLength = 10
v1h = new Float32Array( ( num for num in [0...testLength] ) )
vAd = new JC.VectorD( v1h.length, v1h )
v2h = new Float32Array( ( num for num in [0...testLength] ) )
gpuResult = new Float32Array( testLength )
start = date.getTime()
vBd = new JC.VectorD( testLength )
vBd.copyFrom( testLength, v2h )
vAd.add( vBd )
vAd.copyTo( testLength, gpuResult )
console.log( " GPU #{ testLength } elements vector add used:#{ date.getTime() - start } ms" )
vAd.destroy()
vBd.destroy()

v1h = new Float32Array( ( num for num in [0...testLength] ) )
v2h = new Float32Array( ( num for num in [0...testLength] ) )
start = date.getTime()
for e, i in v2h
  v1h[i] += e
console.log( " CPU #{ testLength } elements vector add used:#{ date.getTime() - start } ms" )

meanError = 0
for e, i in v1h
  meanError += Math.abs( e - gpuResult[i] )
meanError /= v1h.length
console.log( "CPU. vs GPU. Mean absolute error: #{ meanError }, refer to IEEE-754" )

JC.cudaDeviceReset()