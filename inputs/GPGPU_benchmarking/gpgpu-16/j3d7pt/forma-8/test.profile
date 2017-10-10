==28959== NVPROF is profiling process 28959, command: ./test
==28959== Warning: Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==28959== Profiling application: ./test
==28959== Profiling result:
==28959== Metric result:
"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"flop_count_sp","Floating Point Operations(Single Precision)",3171933996,3171933996,3171933996
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"gld_transactions","Global Load Transactions",0,0,0
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"gst_transactions","Global Store Transactions",3229666,3229666,3229666
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"cf_executed","Executed Control-Flow Instructions",28512288,28512288,28512288
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"inst_issued","Instructions Issued",225063589,225063664,225063626
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"shared_load_transactions","Shared Load Transactions",50776572,50776572,50776572
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"shared_store_transactions","Shared Store Transactions",9209172,9209172,9209172
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"achieved_occupancy","Achieved Occupancy",0.496891,0.496895,0.496893
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"stall_sync","Issue Stall Reasons (Synchronization)",29.317691%,29.328361%,29.323026%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"stall_exec_dependency","Issue Stall Reasons (Execution Dependency)",29.555964%,29.576079%,29.566022%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"stall_memory_dependency","Issue Stall Reasons (Data Request)",5.741302%,5.774441%,5.757872%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"issued_ipc","Issued IPC",1.434968,1.436835,1.435901
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"inst_replay_overhead","Instruction Replay Overhead",0.007158,0.007159,0.007159
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"shared_replay_overhead","Shared Memory Replay Overhead",0.000000,0.000000,0.000000
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"global_replay_overhead","Global Memory Replay Overhead",0.006137,0.006137,0.006137
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"gld_efficiency","Global Memory Load Efficiency",0.000000%,0.000000%,0.000000%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"gst_efficiency","Global Memory Store Efficiency",75.037313%,75.037313%,75.037313%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"shared_efficiency","Shared Memory Efficiency",43.296525%,43.296525%,43.296525%
"Tesla K20c (0)","__kernel___forma_kernel__0__(float*, int, int, int, int, int, int, float*)",2,"warp_execution_efficiency","Warp Execution Efficiency",100.000000%,100.000000%,100.000000%
