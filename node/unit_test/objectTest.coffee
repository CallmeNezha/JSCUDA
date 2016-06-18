JC = require("../linear")

date = new Date()
JC.cudaDeviceInit()


batchA = ( new JC.MatrixD( 2, 2, new Float32Array( (num for num in [0...4]) ) ) for i in [0...1] )
batchB = ( new JC.MatrixD( 2, 2, new Float32Array( (num for num in [0...4]) ) ) for i in [0...1] )
batchC = ( new JC.MatrixD( 2, 2, new Float32Array( (num for num in [0...4]) ) ) for i in [0...1] )

matrixWatch = batchC[0]
hostOut = new Float32Array( 4 )
matrixWatch.copyTo( 4, hostOut )
console.log( "Out-put before batch multiply: #{ hostOut }" )

mbdA = new JC.MatrixBatchD( 2, 2, batchA )
mbdB = new JC.MatrixBatchD( 2, 2, batchB )
mbdC = new JC.MatrixBatchD( 2, 2, batchC )

mbdA.multiplyMatrixBatch( mbdB, mbdC )

matrixWatch.copyTo( 4, hostOut )
console.log( "Out-put before batch multiply: #{ hostOut }" )

mbdA.destroy()
mbdB.destroy()
mbdC.destroy()




