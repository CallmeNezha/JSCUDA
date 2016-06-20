JC = require("../linear")

date = new Date()
JC.cudaDeviceInit()


matB = new JC.MatrixD( 2, 2, new Float32Array( [4, 7, 8, 15] ) )

matA = new JC.MatrixD( 2, 2, new Float32Array( [0, 1, 2, 3] ) )

infoArrayh = new Int32Array( 1 )

batchA = new JC.MatrixBatchD( 2, 2, [matA] )
batchB = new JC.MatrixBatchD( 2, 2, [matB] )

JC.linearSolveLUBatched( batchA, batchB, infoArrayh )


hostOut = new Float32Array( 4 )
matB.copyTo( 4, hostOut )
console.log( "Out-put before batch multiply: #{ hostOut }" )
console.log( "Out-put before batch multiply: #{ infoArrayh }" )


batchA.destroy()
batchB.destroy()

matB.destroy()
matA.destroy()

JC.cudaDeviceReset()