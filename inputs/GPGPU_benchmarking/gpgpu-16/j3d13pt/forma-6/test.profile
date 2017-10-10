==8323== NVPROF is profiling process 8323, command: ./test
==8323== Warning: Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==8323== Profiling application: ./test
==8323== Profiling result:
==8323== Metric result:
"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"flop_count_sp","Floating Point Operations(Single Precision)",50258116400,50258116400,50258116400
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"gld_transactions","Global Load Transactions",0,0,0
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"gst_transactions","Global Store Transactions",34192464,34192464,34192464
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"cf_executed","Executed Control-Flow Instructions",547008384,547008384,547008384
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"inst_issued","Instructions Issued",4391961653,4391981947,4391970311
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"shared_load_transactions","Shared Load Transactions",846417408,846417408,846417408
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"shared_store_transactions","Shared Store Transactions",97267584,97267584,97267584
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"achieved_occupancy","Achieved Occupancy",0.998517,0.999162,0.998890
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"stall_sync","Issue Stall Reasons (Synchronization)",25.017429%,25.074871%,25.054088%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",25.420105%,25.459914%,25.445468%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"stall_memory_dependency","Issue Stall Reasons (Data Request)",9.662464%,9.760914%,9.693872%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"issued_ipc","Issued IPC",2.548797,2.552021,2.550399
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"inst_replay_overhead","Instruction Replay Overhead",0.009565,0.009570,0.009567
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"global_replay_overhead","Global Memory Replay Overhead",0.002546,0.002546,0.002546
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"gld_efficiency","Global Memory Load Efficiency",0.000000%,0.000000%,0.000000%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"gst_efficiency","Global Memory Store Efficiency",62.368421%,62.368421%,62.368421%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"shared_efficiency","Shared Memory Efficiency",38.993025%,38.993025%,38.993025%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",6,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
