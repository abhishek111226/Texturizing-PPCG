==19293== NVPROF is profiling process 19293, command: ./test
==19293== Warning: Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==19293== Profiling application: ./test
==19293== Profiling result:
==19293== Metric result:
"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"flop_count_sp","Floating Point Operations(Single Precision)",5172012864,5172012864,5172012864
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"gld_transactions","Global Load Transactions",0,0,0
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"gst_transactions","Global Store Transactions",3576408,3576408,3576408
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"cf_executed","Executed Control-Flow Instructions",27175896,27175896,27175896
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"inst_issued","Instructions Issued",339137735,339137735,339137735
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"shared_load_transactions","Shared Load Transactions",127110432,127110432,127110432
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"shared_store_transactions","Shared Store Transactions",49102112,49102112,49102112
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"achieved_occupancy","Achieved Occupancy",0.248177,0.248177,0.248177
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"stall_sync","Issue Stall Reasons (Synchronization)",11.724353%,11.724353%,11.724353%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",53.760959%,53.760959%,53.760959%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"stall_memory_dependency","Issue Stall Reasons (Data Request)",14.320258%,14.320258%,14.320258%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"issued_ipc","Issued IPC",1.181678,1.181678,1.181678
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"inst_replay_overhead","Instruction Replay Overhead",0.003504,0.003504,0.003504
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"global_replay_overhead","Global Memory Replay Overhead",0.003487,0.003487,0.003487
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"gld_efficiency","Global Memory Load Efficiency",0.000000%,0.000000%,0.000000%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"gst_efficiency","Global Memory Store Efficiency",99.975562%,99.975562%,99.975562%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"shared_efficiency","Shared Memory Efficiency",46.779026%,46.779026%,46.779026%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, float*)",1,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
