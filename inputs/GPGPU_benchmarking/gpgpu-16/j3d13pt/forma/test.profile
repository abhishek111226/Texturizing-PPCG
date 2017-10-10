==21206== NVPROF is profiling process 21206, command: ./test
==21206== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==21206== Profiling application: ./test
==21206== Profiling result:
==21206== Metric result:
"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"flop_count_sp","Floating Point Operations(Single Precision)",25264160800,25264160800,25264160800
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"gld_transactions","Global Load Transactions",0,0,0
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"gst_transactions","Global Store Transactions",11985752,11985752,11985752
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"cf_executed","Executed Control-Flow Instructions",319658752,319658752,319658752
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"inst_issued","Instructions Issued",3226322455,3227709091,3227173467
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"shared_load_transactions","Shared Load Transactions",457541376,457541376,457541376
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"shared_store_transactions","Shared Store Transactions",53189376,53189376,53189376
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"achieved_occupancy","Achieved Occupancy",0.996888,0.997881,0.997336
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"stall_sync","Issue Stall Reasons (Synchronization)",28.053379%,28.239542%,28.125501%
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",27.350308%,27.416716%,27.392462%
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"stall_memory_dependency","Issue Stall Reasons (Data Request)",9.064428%,9.329654%,9.216144%
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"issued_ipc","Issued IPC",2.953314,2.957421,2.955819
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"inst_replay_overhead","Instruction Replay Overhead",0.011436,0.011870,0.011702
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"global_replay_overhead","Global Memory Replay Overhead",0.001199,0.001199,0.001199
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"gld_efficiency","Global Memory Load Efficiency",0.000000%,0.000000%,0.000000%
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"gst_efficiency","Global Memory Store Efficiency",99.603175%,99.603175%,99.603175%
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"shared_efficiency","Shared Memory Efficiency",36.386380%,36.386380%,36.386380%
"Tesla K40c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",3,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
