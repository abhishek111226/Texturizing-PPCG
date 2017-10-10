==27343== NVPROF is profiling process 27343, command: ./test
==27343== Warning: Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==27343== Profiling application: ./test
==27343== Profiling result:
==27343== Metric result:
"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"flop_count_sp","Floating Point Operations(Single Precision)",2538905600,2538905600,2538905600
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_transactions","Global Load Transactions",0,0,0
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_transactions","Global Store Transactions",6209280,6209280,6209280
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"cf_executed","Executed Control-Flow Instructions",28627712,28627712,28627712
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_issued","Instructions Issued",255619355,255620270,255619850
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_load_transactions","Shared Load Transactions",42226688,42226688,42226688
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_store_transactions","Shared Store Transactions",5005568,5005568,5005568
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"achieved_occupancy","Achieved Occupancy",0.870726,0.871241,0.870970
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_sync","Issue Stall Reasons (Synchronization)",25.846039%,25.876284%,25.865466%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",27.313607%,27.323675%,27.318113%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_memory_dependency","Issue Stall Reasons (Data Request)",11.757303%,11.792215%,11.776780%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"issued_ipc","Issued IPC",2.436355,2.439313,2.438147
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_replay_overhead","Instruction Replay Overhead",0.018697,0.018700,0.018699
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"global_replay_overhead","Global Memory Replay Overhead",0.008248,0.008248,0.008248
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_efficiency","Global Memory Load Efficiency",0.000000%,0.000000%,0.000000%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_efficiency","Global Memory Store Efficiency",65.323944%,65.323944%,65.323944%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_efficiency","Shared Memory Efficiency",39.629054%,39.629054%,39.629054%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"flop_count_sp","Floating Point Operations(Single Precision)",2409976800,2409976800,2409976800
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_transactions","Global Load Transactions",29882160,29882160,29882160
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_transactions","Global Store Transactions",4374720,4374720,4374720
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"cf_executed","Executed Control-Flow Instructions",34498304,34498304,34498304
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_issued","Instructions Issued",323336374,323709255,323482258
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_load_transactions","Shared Load Transactions",42226688,42226688,42226688
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_store_transactions","Shared Store Transactions",8041216,8041216,8041216
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"achieved_occupancy","Achieved Occupancy",0.872900,0.873138,0.873055
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_sync","Issue Stall Reasons (Synchronization)",30.060843%,30.166589%,30.114969%
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",16.155983%,16.236247%,16.201839%
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_memory_dependency","Issue Stall Reasons (Data Request)",28.952361%,29.112181%,29.029505%
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"issued_ipc","Issued IPC",1.341205,1.351557,1.347000
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_replay_overhead","Instruction Replay Overhead",0.137001,0.138312,0.137514
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"global_replay_overhead","Global Memory Replay Overhead",0.059276,0.059276,0.059276
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_efficiency","Global Memory Load Efficiency",50.000000%,50.000000%,50.000000%
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_efficiency","Global Memory Store Efficiency",78.040386%,78.040386%,78.040386%
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_efficiency","Shared Memory Efficiency",35.183058%,35.183058%,35.183058%
"Tesla K20c (0)","__kernel___forma_kernel__1__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"flop_count_sp","Floating Point Operations(Single Precision)",2396979200,2396979200,2396979200
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_transactions","Global Load Transactions",8171520,8171520,8171520
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_transactions","Global Store Transactions",6128640,6128640,6128640
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"cf_executed","Executed Control-Flow Instructions",33607680,33607680,33607680
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_issued","Instructions Issued",311560242,311569766,311563479
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_load_transactions","Shared Load Transactions",40857600,40857600,40857600
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_store_transactions","Shared Store Transactions",5204574,5204642,5204608
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"achieved_occupancy","Achieved Occupancy",0.873325,0.873576,0.873425
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_sync","Issue Stall Reasons (Synchronization)",22.307464%,22.353618%,22.334120%
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",23.600867%,23.608710%,23.606014%
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_memory_dependency","Issue Stall Reasons (Data Request)",28.025722%,28.058086%,28.044695%
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"issued_ipc","Issued IPC",1.791479,1.792691,1.791922
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_replay_overhead","Instruction Replay Overhead",0.060898,0.060930,0.060909
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"global_replay_overhead","Global Memory Replay Overhead",0.016231,0.016231,0.016231
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_efficiency","Global Memory Load Efficiency",81.250000%,81.250000%,81.250000%
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_efficiency","Global Memory Store Efficiency",57.142857%,57.142857%,57.142857%
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_efficiency","Shared Memory Efficiency",38.099127%,38.099183%,38.099155%
"Tesla K20c (0)","__kernel___forma_kernel__2__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"flop_count_sp","Floating Point Operations(Single Precision)",2485929600,2485929600,2485929600
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_transactions","Global Load Transactions",0,0,0
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_transactions","Global Store Transactions",3217536,3217536,3217536
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"cf_executed","Executed Control-Flow Instructions",43620480,43620480,43620480
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_issued","Instructions Issued",288046365,288046448,288046400
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_load_transactions","Shared Load Transactions",40857600,40857600,40857600
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_store_transactions","Shared Store Transactions",7655936,7655936,7655936
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"achieved_occupancy","Achieved Occupancy",0.873467,0.873672,0.873578
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_sync","Issue Stall Reasons (Synchronization)",29.256044%,29.525105%,29.419438%
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",15.944406%,16.108197%,16.023202%
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"stall_memory_dependency","Issue Stall Reasons (Data Request)",35.912254%,36.011744%,35.949192%
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"issued_ipc","Issued IPC",1.543277,1.547176,1.544942
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"inst_replay_overhead","Instruction Replay Overhead",0.005677,0.005678,0.005678
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"global_replay_overhead","Global Memory Replay Overhead",0.005528,0.005528,0.005528
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gld_efficiency","Global Memory Load Efficiency",0.000000%,0.000000%,0.000000%
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"gst_efficiency","Global Memory Store Efficiency",87.401575%,87.401575%,87.401575%
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"shared_efficiency","Shared Memory Efficiency",36.932861%,36.932861%,36.932861%
"Tesla K20c (0)","__kernel___forma_kernel__3__(float*, int, int, int, float*, float*, float*, int, int, int, float*)",3,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
