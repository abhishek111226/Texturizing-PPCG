/*
 * Copyright 2016
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Abhishek Patwardhan
 * IIT Hyderabad, India
 *	
 */

#include <isl/aff.h>
#include <string.h>
#include <stdbool.h>
#include "gpu.h"
#include "print.h"
#include "texture.h"
#include <pet/tree.h>
#include <pet/include/pet.h>



void print_analysis(struct kernel_analysis * kernel_analysis)
{
	int i,j;
	struct candidate_info * analysis_tableu = kernel_analysis->analysis_tableu;
	printf("\n\n");
	printf("\n\t ---- Printing results of static analysis ------");
	for(i=0;i<kernel_analysis->candidates_cnt;i++)
	{
		printf("\n\t Analysis for array %s",analysis_tableu[i].array_info->name);
		for(j=0;j<kernel_analysis->kernel_cnt;j++)
		{
			printf("\n\t\t\t After kernel %d : %d",j,analysis_tableu[i].latest_access[j]);
		}
		printf("\n");	
	}
	printf("\n\t ---- End ------");	
	printf("\n\n");
}

bool print_device_arrays_or_not(struct gpu_array_info * array)
{
	printf("\n checking for %s",array->name);
	if(array->linearize)
		return true;
	if(is_ccp_greater_than_two && (array->is_in_texture || array->is_in_surface))
		return false;	
	if(!is_ccp_greater_than_two && array->is_in_texture && array->is_read_only)
		return false;
	return true;
}

/*int get_index_for_candidate_array(char * array_name)
{
	int i;
	for(i=0;i<cand_info.n_texture_candidates;i++)
	{
		if(!strcmp(cand_info.candidates[i]->array_name,array_name))
			return i;
	}
	return -1;
}*/

/*
void update_types_for_array(char* name,char* type)
{
	struct surf_read_access* t =  home_surf_read_access;
	while(t!=NULL)
	{
		const char * tmp = extract_array_name_from_access(t->access_expr);
		if(!strcmp(tmp,name))
		{
			t->type = type;		
		}
		t=t->next;
	}
}

void surface_temp_decide_types(int n_array,struct gpu_array_info * arrays)
{
	int i;
	for(i=0;i<n_array;i++)
	{
		if(arrays[i].is_in_surface)
		{
			update_types_for_array(arrays[i].name,arrays[i].type);	
		}		
	}
}
*/


/*void insert_tex_decl(struct gpu_array_info*  arr)
{
	printf("\n inserting %s \n",arr->name);
	if(home_tex_decl==NULL){
		home_tex_decl = (struct tex_decl*) malloc(sizeof(struct tex_decl));
		home_tex_decl->tex_array=arr;
		home_tex_decl->latest_access=none;
		home_tex_decl->next=NULL;
	}
	else{
		struct tex_decl * t= home_tex_decl;
		while(t->next!=NULL){
			if(!strcmp(t->tex_array->name,arr->name))
				break;
			else
				t=t->next;	
		}
		if(!strcmp(t->tex_array->name,arr->name))
			return ;
		else{
			t ->next = (struct tex_decl*) malloc(sizeof(struct tex_decl));
			t->next->tex_array=arr;
			t->next->latest_access=none;
			t->next->next=NULL;
		}

	}

}
*/
__isl_give isl_printer *bind_linear_texture(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
		// generate statement like cudaBindTexture(NULL,tex_Ta, ga, sizeof(int)* VEC_SIZE);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaBindTexture(NULL,");
		p = isl_printer_print_str(p,"texRef_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", ");
		p = isl_printer_print_str(p,"dev_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", ");
		p = gpu_array_info_print_size(p, array);
		p = isl_printer_print_str(p, ");");
		p = isl_printer_end_line(p);
		return p;
}

__isl_give isl_printer *print_dimensions_as_args(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
	int i;
  	for (i = array->n_index -1; i >=0 ; i--) 
	{
		if(i!=array->n_index -1)
			p = isl_printer_print_str(p,", ");
                isl_ast_expr *bound;
                bound = isl_ast_expr_get_op_arg(array->bound_expr, 1 + i);
	 	p = isl_printer_print_ast_expr(p, bound);
                isl_ast_expr_free(bound);
        }
	return p;

}
__isl_give isl_printer *print_linearized_dimensions(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
	int i;
  	for (i = array->n_index -1; i >=0 ; i--) 
	{
		if(i!=array->n_index -1)
			p = isl_printer_print_str(p,"* ");
                isl_ast_expr *bound;
                bound = isl_ast_expr_get_op_arg(array->bound_expr, 1 + i);
	 	p = isl_printer_print_ast_expr(p, bound);
                isl_ast_expr_free(bound);
        }
	return p;

}


/*
__isl_give isl_printer *print_surface_read_to_temp(__isl_take isl_printer *p, struct ppcg_kernel *kernel)
{
	struct surf_read_expressions *next, * t = kernel->decl;
	printf("print surface read to temp called");	
	if(t==NULL)
		printf("------ NULL ---------");	
	while(t!=NULL)
	{
		printf("\n surface access read printed for %s",t->type);
		p = print_surface_read(p,t->temp_name, t->type,  t->access_expr);
		next= t->next;
		isl_ast_expr_free(t->access_expr);
		free(t->temp_name);
		free(t);
		t= next;
	}
	//free(kernel->decl);
	printf("Printing done");
	return p;
}
*/
__isl_give isl_printer *declare_channel_format(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaChannelFormatDesc channelDesc_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,"= cudaCreateChannelDesc<");
		p = isl_printer_print_str(p,array->type);
		p = isl_printer_print_str(p,">();");
		p = isl_printer_end_line(p);


}


__isl_give isl_printer *malloc_linearized_array(__isl_take isl_printer *p, struct gpu_array_info*  array, bool is_in_surface)
{
		//cudaMallocArray(&arr, &channelDesc, width, height);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaMallocArray(&cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,", &channelDesc_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,", ");
		p = print_linearized_dimensions(p,array);
		p = isl_printer_print_str(p,", 1");			
		if(is_in_surface)
			p = isl_printer_print_str(p,", cudaArraySurfaceLoadStore");
		p = isl_printer_print_str(p,");");
		p = isl_printer_end_line(p);
}


__isl_give isl_printer *declare_cuda_array(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
		// cudaArray* arr;
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaArray* cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ";");
		p = isl_printer_end_line(p);

}
__isl_give isl_printer *malloc_2d_array(__isl_take isl_printer *p, struct gpu_array_info*  array, bool is_in_surface)
{
		//cudaMallocArray(&arr, &channelDesc, width, height);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaMallocArray(&cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,", &channelDesc_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,", ");
		p = print_dimensions_as_args(p,array);
		if(array->n_index==1)
			p = isl_printer_print_str(p,", 1");			
		if(is_in_surface)
			p = isl_printer_print_str(p,", cudaArraySurfaceLoadStore");
		p = isl_printer_print_str(p,");");
		p = isl_printer_end_line(p);
}
__isl_give isl_printer *memcpy_2d_array(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "cudaMemcpyToArray(cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", 0, 0, ");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", ");
		p = gpu_array_info_print_size(p, array);
		p = isl_printer_print_str(p, ", ");
		p = isl_printer_print_str(p,"cudaMemcpyHostToDevice);");
		p = isl_printer_end_line(p);
}

__isl_give isl_printer *memcpy_2d_array_from_device(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "cudaMemcpyFromArray(");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", 0, 0, ");
		p = gpu_array_info_print_size(p, array);
		p = isl_printer_print_str(p, ", ");
		p = isl_printer_print_str(p," cudaMemcpyDeviceToHost);");
		p = isl_printer_end_line(p);
}

__isl_give isl_printer *copy_2d_array_device_to_device(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
	// cudaMemcpyToArray(cuArr_tex_B, 0, 0, tex_B, (4) * (4) * sizeof(int), cudaMemcpyHostToDevice);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaMemcpyToArray(cuArr_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", 0, 0, ");
	p = isl_printer_print_str(p, "dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = gpu_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p,"cudaMemcpyDeviceToDevice);");
	p = isl_printer_end_line(p);
	return p;
}



__isl_give isl_printer *bind_cu_array_to_texture(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "cudaBindTextureToArray(texRef_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", channelDesc_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ");");
		p = isl_printer_end_line(p);
}
__isl_give isl_printer *bind_cu_array_to_surface(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
		//cudaBindSurfaceToArray(surfD, d_arr,Channeldesc)
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "cudaBindSurfaceToArray(surfRef_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", channelDesc_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ");");
		p = isl_printer_end_line(p);
}

__isl_give isl_printer *declare_cuda_extent(__isl_take isl_printer *p, struct gpu_array_info*  array)
{
		// const cudaExtent extent = make_cudaExtent(4,4,4);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"const cudaExtent extent_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,"= make_cudaExtent(");
		print_dimensions_as_args(p,array);
		p = isl_printer_print_str(p, ");");
		p = isl_printer_end_line(p);
}
__isl_give isl_printer *malloc_3d_cu_array(__isl_take isl_printer *p, struct gpu_array_info*  array, bool is_in_surface)
{
	//cudaMalloc3DArray(&cuArr_tex_A,&channelDesc_tex_A,extent_tex_A);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaMalloc3DArray(&cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,", &channelDesc_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,", extent_");
		p = isl_printer_print_str(p, array->name);
		if(is_in_surface)
			p = isl_printer_print_str(p,", cudaArraySurfaceLoadStore");
		p = isl_printer_print_str(p, ");");
		p = isl_printer_end_line(p);
}

__isl_give isl_printer * handle_3d_copy_between_host_to_cu_array(__isl_take isl_printer *p, struct gpu_array_info*  array , enum data_transfer direction)
{
	/*cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_data, extent.width*sizeof(float), extent.width, extent.height);
	copyParams.dstArray = arr;
	copyParams.extent   = extent;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams); */

		char * prefix = "copyParams_";
		char * postfix;
		if(direction==to_device)
			postfix = "_to_device";
		else if(direction==from_device)
			postfix = "_from_device";
		else if(direction==among_device)
			postfix = "_among_device";
		char * copy_param_name = (char *)malloc((strlen(prefix)+strlen(array->name)+strlen(postfix)+1)*sizeof(char));

		strcpy(copy_param_name,prefix);
		strcat(copy_param_name,array->name);
		strcat(copy_param_name, postfix);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaMemcpy3DParms ");
		p = isl_printer_print_str(p,copy_param_name);
		
		p = isl_printer_print_str(p,"= {0}");
		p = isl_printer_print_str(p, ";");
		p = isl_printer_end_line(p);
		
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,copy_param_name);
		if(direction==to_device)
			p = isl_printer_print_str(p,".srcPtr ");
		else if(direction==from_device)
			p = isl_printer_print_str(p,".dstPtr ");
		else if(direction==among_device)
			p = isl_printer_print_str(p,".srcPtr ");

		p = isl_printer_print_str(p,"= make_cudaPitchedPtr((void*)");
		if(direction==among_device)
			p = isl_printer_print_str(p, "dev_");				
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ", extent_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ".width*sizeof(");
		p = isl_printer_print_str(p, array->type);
		p = isl_printer_print_str(p, "), extent_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ".width, ");
		p = isl_printer_print_str(p, "extent_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p, ".height);");
		p = isl_printer_end_line(p);

		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,copy_param_name);
		if(direction==to_device)
			p = isl_printer_print_str(p,".dstArray = cuArr_");
		else if(direction==from_device)
			p = isl_printer_print_str(p,".srcArray = cuArr_");
		else if(direction==among_device)
			p = isl_printer_print_str(p,".dstArray = cuArr_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,";");
		p = isl_printer_end_line(p);

		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,copy_param_name);
		p = isl_printer_print_str(p,".extent = extent_");
		p = isl_printer_print_str(p, array->name);
		p = isl_printer_print_str(p,";");
		p = isl_printer_end_line(p);

		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,copy_param_name);
		if(direction==to_device)
			p = isl_printer_print_str(p,".kind = cudaMemcpyHostToDevice");
		else if(direction==from_device)
			p = isl_printer_print_str(p,".kind = cudaMemcpyDeviceToHost");
		else if(direction==among_device)
			p = isl_printer_print_str(p,".kind = cudaMemcpyDeviceToDevice");

		p = isl_printer_print_str(p,";");
		p = isl_printer_end_line(p);

		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,"cudaMemcpy3D(&");
		p = isl_printer_print_str(p, copy_param_name);
		p = isl_printer_print_str(p,");");
		p = isl_printer_end_line(p);


		free(copy_param_name);
		return p;
}

__isl_give isl_printer * copy_data_from_device_to_device(__isl_take isl_printer * p,struct ppcg_kernel *kernel)
{
	
	int i;
	if(is_ccp_greater_than_two)
	 return p ;
	for(i=0;i<kernel->n_array;i++)
	{
		if(kernel->array[i].is_written && kernel->array[i].array->is_in_texture)
		{
			if(kernel->array[i].array->n_index<3)
				p = copy_2d_array_device_to_device(p,kernel->array[i].array);
			else
				p = handle_3d_copy_between_host_to_cu_array(p,kernel->array[i].array,among_device);
		}
	}	
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}


__isl_give isl_printer * copy_cuda_array_from_device(__isl_take isl_printer *p, struct gpu_array_info *array)
{

		if(array->n_index<3)
			p = memcpy_2d_array_from_device(p,array);	
		else
		{
			printf("--------------3d copy for %s %d",array->name,array->n_index);
			p = handle_3d_copy_between_host_to_cu_array(p,array,from_device);
		}
		p = isl_printer_start_line(p);
		p = isl_printer_end_line(p);
		return p;
}

__isl_give isl_printer *bind_texture_or_surface(__isl_take isl_printer *p, struct gpu_array_info*  array, bool is_surface_to_bind)
{
		p = isl_printer_start_line(p);
		p = isl_printer_end_line(p);
		if(array->linearize)
		{
			p = bind_linear_texture(p,array);					
		}
		else if(array->n_index==1)
		{
			p = declare_cuda_array(p,array);	
			p = declare_channel_format(p,array);
			p = malloc_linearized_array(p,array,is_surface_to_bind);
			p = memcpy_2d_array(p,array);
			if(is_surface_to_bind)
				p = bind_cu_array_to_surface(p,array);
			else
				p = bind_cu_array_to_texture(p,array);
		}
		else if(array->n_index==2)
		{
			p = declare_cuda_array(p,array);
			p = declare_channel_format(p,array); 
			p = malloc_2d_array(p,array,is_surface_to_bind);
			p = memcpy_2d_array(p,array);
			if(is_surface_to_bind)
				p = bind_cu_array_to_surface(p,array);
			else
				p = bind_cu_array_to_texture(p,array);
		}
		else if(array->n_index==3)
		{
			p = declare_cuda_array(p,array);
			p = declare_channel_format(p,array); 
			p = declare_cuda_extent(p,array);
			p = malloc_3d_cu_array(p,array,is_surface_to_bind);
			p = handle_3d_copy_between_host_to_cu_array(p,array,to_device);
			if(is_surface_to_bind)
				p = bind_cu_array_to_surface(p,array);
			else
				p = bind_cu_array_to_texture(p,array);
		}	
		return p;
}


/*isl_printer * print_surf_reads_to_temp(isl_printer *p)
{
	struct surf_read_access* t =  home_surf_read_access;
	while(t!=NULL)
	{
			p = isl_printer_start_line(p);
			p = isl_printer_print_str(p,"\t texture< ");
			p = isl_printer_print_str(p, t->tex_array->type);
			p = isl_printer_print_str(p, ", ");
			if(!t->tex_array->lintr1earize)
				p = isl_printer_print_int(p, t->tex_array->n_index);
			else
				p = isl_printer_print_int(p, 1);
			p = isl_printer_print_str(p, ", ");
			p = isl_printer_print_str(p, "cudaReadModeElementType");
			p = isl_printer_print_str(p, " > ");
			p = isl_printer_print_str(p, "texRef_");
			p = isl_printer_print_str(p, t->tex_array->name);
			p = isl_printer_print_str(p, ";");
			p = isl_printer_end_line(p); 
			t=t->next;	
	} 		 

}*/

__isl_give isl_printer *  print_texture_declaration(__isl_take isl_printer *p, struct gpu_array_info * array, bool is_extern)
{
	p = isl_printer_start_line(p);
	if(is_extern)
		p = isl_printer_print_str(p,"\t extern texture< ");
	else
		p = isl_printer_print_str(p,"\t texture< ");
	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, ", ");
	if(!array->linearize)
		p = isl_printer_print_int(p, array->n_index);
	else
		p = isl_printer_print_int(p, 1);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, "cudaReadModeElementType");
	p = isl_printer_print_str(p, " > ");
	p = isl_printer_print_str(p, "texRef_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
}

__isl_give isl_printer *  print_surface_declaration(__isl_take isl_printer *p, struct gpu_array_info * array, bool is_extern)
{
	//surface<void, 2> inputSurfRef;
	p = isl_printer_start_line(p);
	if(is_extern)
		p = isl_printer_print_str(p,"\t extern surface< ");
	else
		p = isl_printer_print_str(p,"\t surface< ");
	p = isl_printer_print_str(p, "void");
	p = isl_printer_print_str(p, ", ");
	if(!array->linearize)
		p = isl_printer_print_int(p, array->n_index);
	else
		p = isl_printer_print_int(p, 1);
	p = isl_printer_print_str(p, " > ");
	p = isl_printer_print_str(p, "surfRef_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
}

isl_printer * print_texture_or_surface_decl(isl_printer *p, struct gpu_prog * prog)
{
	int i;
	p=isl_printer_print_str(p,"\n#ifdef HOSTCODE \n ");
	for(i=0;i<prog->n_array;i++)
	{
		if(prog->array[i].is_in_texture)
			p = print_texture_declaration(p,&prog->array[i],false);
		else if (prog->array[i].is_in_surface)
			p = print_surface_declaration(p,&prog->array[i],false);
	} 
	p=isl_printer_print_str(p,"\n#elif DEVICECODE \n");
	for(i=0;i<prog->n_array;i++)
	{
		if(prog->array[i].is_in_texture)
			p = print_texture_declaration(p,&prog->array[i],true);
		else if (prog->array[i].is_in_surface)
			p = print_surface_declaration(p,&prog->array[i],true);
	} 
	p=isl_printer_print_str(p,"\n #endif \n");
	return p;
}
__isl_give isl_printer *bind_device_textures_surfaces(__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;
	for (i = 0; i < prog->n_array; ++i) 
	{
		if(prog->array[i].is_in_texture)
		{
			bind_texture_or_surface(p,&prog->array[i],false);
		}	
		else if(prog->array[i].is_in_surface)
		{
			bind_texture_or_surface(p,&prog->array[i],true);
		}
	}

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	return p;
}

__isl_give isl_printer *free_cuda_array(__isl_take isl_printer *p,struct gpu_prog *prog)
{
	int i;
	for (i = 0; i < prog->n_array; ++i) 
	{
		if((prog->array[i].is_in_texture || prog->array[i].is_in_surface) && !prog->array[i].linearize)
		{	
			p = isl_printer_start_line(p);
			p = isl_printer_print_str(p,"cudaFreeArray");
			p = isl_printer_print_str(p,"(");
			p = isl_printer_print_str(p,"cuArr_");
			p = isl_printer_print_str(p, prog->array[i].name);
			p = isl_printer_print_str(p, ");");
			p = isl_printer_end_line(p);
		}
	}
	return p;
}




__isl_give isl_printer *unbind_device_textures_surfaces(__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;
	p = isl_printer_start_line(p);
 	p = isl_printer_end_line(p);
	for (i = 0; i < prog->n_array; ++i) 
	{
		if(prog->array[i].is_in_texture) 
		{
			//generate statement like cudaUnbindTexture(tex_Ta);
		 	p = isl_printer_start_line(p);
			p = isl_printer_print_str(p,"cudaUnbindTexture(");
			p = isl_printer_print_str(p,"texRef_");
			p = isl_printer_print_str(p, prog->array[i].name);
			p = isl_printer_print_str(p, ");");
			p = isl_printer_end_line(p);  
		}
		
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}






/*
__isl_give isl_printer *declare_device_textures(__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;
	for (i = 0; i < prog->n_array; ++i) 
	{
		/* FIXME As of now assuming that arrays are declared and allocated memory using cudamalloc.
		if (!gpu_array_requires_device_allocation(&prog->array[i]))
			continue;
			
		    TODO cudaReadModeElementType, type checking 	
		*/
		// generate statement like texture<int,1,cudaReadModeElementType> tex_Ta;
/*		if(prog->array[i].is_in_texture) 
		{		
			p = isl_printer_start_line(p);
			p = isl_printer_print_str(p,"extern texture<");
			p = isl_printer_print_str(p, prog->array[i].type);
			p = isl_printer_print_str(p, ", ");
			char dimchar[sizeof(int)];
			sprintf(dimchar, "%d",prog->array[i].n_index);
			p = isl_printer_print_str(p, dimchar);
			p = isl_printer_print_str(p, ", ");
			p = isl_printer_print_str(p, "cudaReadModeElementType");
			p = isl_printer_print_str(p, "> ");
			p = isl_printer_print_str(p, "texRef_");
			p = isl_printer_print_str(p, prog->array[i].name);
			p = isl_printer_print_str(p, ";");
			p = isl_printer_end_line(p);
			
		}
		
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}


__isl_give isl_printer * declare_device_textures_for_hostfile(__isl_take isl_printer *p,struct gpu_prog *prog)
{
	int i;
	for (i = 0; i < prog->n_array; ++i) 
	{
		/* FIXME As of now assuming that arrays are declared and allocated memory using cudamalloc.
		if (!gpu_array_requires_device_allocation(&prog->array[i]))
			continue;
			
		    TODO cudaReadModeElementType, type checking 	
		*/
		// generate statement like texture<int,1,cudaReadModeElementType> tex_Ta;
/*		if(prog->array[i].is_in_texture) 
		{		
			p = isl_printer_start_line(p);
			p = isl_printer_print_str(p,"texture<");
			p = isl_printer_print_str(p, prog->array[i].type);
			p = isl_printer_print_str(p, ", ");
			char dimchar[sizeof(int)];
			sprintf(dimchar, "%d",prog->array[i].n_index);
			p = isl_printer_print_str(p, dimchar);
			p = isl_printer_print_str(p, ", ");
			p = isl_printer_print_str(p, "cudaReadModeElementType");
			p = isl_printer_print_str(p, "> ");
			p = isl_printer_print_str(p, "texRef_");
			p = isl_printer_print_str(p, prog->array[i].name);
			p = isl_printer_print_str(p, ";");
			p = isl_printer_end_line(p);
			insert_tex_decl(&prog->array[i]);
		}
		
	}
	//p = isl_printer_start_line(p);
	//p = isl_printer_end_line(p);
	return p;
}
*/
/*__isl_give isl_printer *print_texture_access(__isl_take isl_printer *p,__isl_keep isl_ast_expr *expr)
{
	int i = 0;

	p = isl_printer_print_ast_expr(p, expr->u.op.args[0]);
	for (i = 1; i < expr->u.op.n_arg; ++i) {
		p = isl_printer_print_str(p, "[");
		p = isl_printer_print_ast_expr(p, expr->u.op.args[i]);
		p = isl_printer_print_str(p, "]");
	}

	return p;
}

*/

/*void update_analysis(const char *array_name, isl_bool is_access_write)
{
	
	if(home_tex_decl)
	{
		struct tex_decl * t= home_tex_decl;
		while(t!=NULL){
			if(!strcmp(t->tex_array->name,array_name))
			{
				enum RWbar updated ; 
				if(is_access_write==isl_bool_true)
				{
					updated = 
					if(!t->tex_array->is_written)
						t->tex_array->is_written=true;
				}	
				else
				{
					updated = get_updated_state(t->latest_access,false);
				}
				t->latest_access = updated;
				return ;
			}
			else
				t=t->next;	
		}
	}
	// ERROR CASE HERE		
}
*/	

void clear_texture_flags(struct gpu_prog * prog)
{
	int i;
	for(i=0;i<prog->n_array;i++)
	{
		prog->array[i].is_in_texture=false;
		prog->array[i].is_in_surface=false;
		prog->array[i].is_read_only=false;
		prog->array[i].is_write_only=false;
	}
} 

void gentemp( char ** temp, isl_ast_expr * expr, int temp_cnt)
{
	isl_ctx* ctx;
	int i=0;
	char * temp_cnt_char;
	int size= asprintf(&temp_cnt_char, "%d", temp_cnt);
	ctx= isl_ast_expr_get_ctx(expr);
	isl_printer * new_var_name = isl_printer_to_str(ctx);
	new_var_name=isl_printer_print_ast_expr(new_var_name,expr);
	char * d = isl_printer_get_str(new_var_name);
	//printf("\n original access : %s",d);	
	char * s = "surf_temp_";
	*temp = (char *)malloc(sizeof(char)*(1+strlen(s)+strlen(d)+strlen(temp_cnt_char)));
	strcpy(*temp,s);
	strcat(*temp,d);
	strcat(*temp,"_");
	strcat(*temp,temp_cnt_char);
	s = *temp;	
	for(i=0;i<strlen(s);i++)
	{
		if(s[i]=='[' || s[i]==']')
			s[i]='B';
		else if(s[i]=='(' || s[i]==')')
			s[i]='C';
		else if(s[i]=='+')
			s[i]='a';
		else if(s[i]=='-')
			s[i]='s';
		else if(s[i]=='*')
			s[i]='m';
		else if(s[i]=='/')
			s[i]='d';
		else if(s[i]=='%')
			s[i]='m';	
		else if(s[i]==' ')
			s[i]='w';	
	}	
	//strcpy(*temp,s);
	//printf(" \n %s",*temp);
	isl_printer_free(new_var_name);	
	free(temp_cnt_char);
}




void insert_surf_read_access(__isl_keep isl_ast_expr * expr, struct surf_read_expressions ** read_surf_list,__isl_keep isl_ast_expr * expr_old, char * type, char * temp_name)
{
	if(*read_surf_list==NULL){
		*read_surf_list = (struct surf_read_expressions*) malloc(sizeof(struct surf_read_expressions));
		(*read_surf_list)->surface_read_expr = expr;
		(*read_surf_list)->access_expr = expr_old;
		//gentemp(&((*read_surf_list)->temp_name),expr);
		(*read_surf_list)->type = type;
		(*read_surf_list)->temp_name = temp_name;
		(*read_surf_list)->next=NULL;	
		//return (*read_surf_list)->temp_name ;
	}
	else{
		struct surf_read_expressions * t= *read_surf_list;
		while(t->next!=NULL){
			if(isl_ast_expr_is_equal(t->surface_read_expr,expr)==isl_bool_true)
			{
				isl_ast_expr_free(expr);
				return ;
			}
			else
				t=t->next;	
		}
		if(isl_ast_expr_trees_compare(t->surface_read_expr,expr)==isl_bool_true)
		{
			isl_ast_expr_free(expr);
			return ;
		}		
		else{
			t ->next = (struct surf_read_expressions*) malloc(sizeof(struct surf_read_expressions));
			t->next->surface_read_expr=expr;
			t->next->access_expr=expr_old;
			t->next->type=type;
			t->next->temp_name=temp_name;
			//gentemp(&t->next->temp_name,expr);
			t->next->next=NULL;
			return ;
		}
	}
}


int pet_expr_modify_access_to_var(__isl_keep isl_ast_expr *expr,char *type, struct surf_read_expressions ** read_surf_list, int temp_cnt)
{
	//char * temp = insert_surf_read_access(expr,type,read_surf_list);
	char * temp;	
	gentemp(&temp,expr,temp_cnt++);	
	isl_ast_expr* t_expr=  construct_surface_read_access(expr,type,temp);
	alter_access(expr,temp);
	insert_surf_read_access(t_expr,read_surf_list,expr,type,temp);
	printf("\ninserting done ");
	fflush(stdout);
	return temp_cnt;

}

void decide_texture_profitability(struct kernel_analysis * kernel_analysis)
{
	int i,j;
	struct candidate_info * analysis_tableu = kernel_analysis->analysis_tableu;
	printf("\n\n");
	printf("\n\t ---- Printing results of cost model ------");
	for(i=0;i<kernel_analysis->candidates_cnt;i++)
	{
		if(analysis_tableu[i].array_info->is_in_texture || analysis_tableu[i].array_info->is_in_surface)
		{
			int avg_ideal_warp_size = analysis_tableu[i].array_info->texture_cost_ideal_warp_size; 
			if(analysis_tableu[i].array_info->texture_cost_ref_counter)			
				avg_ideal_warp_size = (int) avg_ideal_warp_size/analysis_tableu[i].array_info->texture_cost_ref_counter; 	
			else
				avg_ideal_warp_size = (int) avg_ideal_warp_size;	
			if(avg_ideal_warp_size > 65 || analysis_tableu[i].array_info->is_write_only || analysis_tableu[i].array_info->n_index <= 1) 	
			{
				printf("\n Unprofitable \t %s \t == %d \t",analysis_tableu[i].array_info->name, avg_ideal_warp_size);	
				analysis_tableu[i].array_info->is_in_texture = false;
				analysis_tableu[i].array_info->is_in_surface = false;
			}
			else if (analysis_tableu[i].array_info->is_in_texture)
			{
				printf("\n Profitable \t %s \t == %d \t",analysis_tableu[i].array_info->name, avg_ideal_warp_size);
			}
			else if (analysis_tableu[i].array_info->is_in_surface && avg_ideal_warp_size > 50)
			{
printf("\n Unprofitable \t %s \t == %d \t",analysis_tableu[i].array_info->name, avg_ideal_warp_size);	
				analysis_tableu[i].array_info->is_in_texture = false;
				analysis_tableu[i].array_info->is_in_surface = false;

			}
			else if(analysis_tableu[i].array_info->is_in_surface && avg_ideal_warp_size < 5)
			{
printf("\n Unprofitable \t %s \t == %d \t",analysis_tableu[i].array_info->name, avg_ideal_warp_size);	
				analysis_tableu[i].array_info->is_in_texture = false;
				analysis_tableu[i].array_info->is_in_surface = false;
			}
			else
			{
printf("\n Profitable \t %s \t == %d \t",analysis_tableu[i].array_info->name, avg_ideal_warp_size);
			} 				
		}
	}
	printf("\n\t ---- End ------");	
	printf("\n\n");
}



int annotate_node(__isl_keep pet_expr *expr,isl_id_to_ast_expr *ref2expr,struct candidate_info * candidate, bool is_access_write, struct surf_read_expressions ** read_surf_list, int temp_cnt)
{
	if(candidate->array_info->is_in_texture)
	{
		if(!is_access_write)
		{
			//pet_expr_mark_texture_access(expr);
			isl_ast_expr *ast_expr;
			ast_expr = isl_id_to_ast_expr_get(ref2expr,pet_expr_access_get_ref_id(expr));
			isl_ast_expr_modify_access_to_tex_call(ast_expr,candidate->array_info->linearize);
			isl_ast_expr_free(ast_expr);
			//expr->type=pet_expr_call;
			return -1;		
		}

	}
	else if(candidate->array_info->is_in_surface)
	{
		pet_expr_set_array_data_type(expr,candidate->array_info->type);
		if(is_access_write)
		{
			pet_expr_mark_surface_access(expr);
			return -1;
		}
		else
		{
			isl_ast_expr *ast_expr;
			ast_expr = isl_id_to_ast_expr_get(ref2expr,pet_expr_access_get_ref_id(expr));
			temp_cnt = pet_expr_modify_access_to_var(ast_expr,candidate->array_info->type,read_surf_list,temp_cnt);
			isl_ast_expr_free(ast_expr);
			return temp_cnt;
		}
	}
	return -1;
}


void decide_surface_or_texture(struct kernel_analysis * kernel_analysis)
{
	int i,j;
	struct candidate_info * analysis_tableu = kernel_analysis->analysis_tableu;
	bool is_read_only;
	bool is_marked_invalid;
	for(i=0;i<kernel_analysis->candidates_cnt;i++)
	{
		is_read_only = true;
		is_marked_invalid = false;
		for(j=0;j<kernel_analysis->kernel_cnt;j++)
		{
			if(analysis_tableu[i].latest_access[j]==invalid)
			{
				analysis_tableu[i].array_info->is_in_texture=false;
				analysis_tableu[i].array_info->is_in_surface=false;
				is_marked_invalid = true;				
				break;
			}
			else if(analysis_tableu[i].latest_access[j]==write)
				is_read_only=false;
		}
		if(is_marked_invalid)
			continue;
		if(is_read_only)
		{
			analysis_tableu[i].array_info->is_in_texture=true;
			analysis_tableu[i].array_info->is_in_surface=false;
		}
		else
		{
			if(analysis_tableu[i].array_info->linearize)
			{
				analysis_tableu[i].array_info->is_in_texture=true;
				analysis_tableu[i].array_info->is_in_surface=false;
			}
			else if(is_ccp_greater_than_two)
			{
				analysis_tableu[i].array_info->is_in_texture=false;
				analysis_tableu[i].array_info->is_in_surface=true;
			}
			else
			{
				analysis_tableu[i].array_info->is_in_texture=false;
				analysis_tableu[i].array_info->is_in_surface=false;
			}
		}
		analysis_tableu[i].array_info->is_read_only=is_read_only;

	}
	printf("\n\t\t ------ Decision of textur or surface -------------");
	for(i=0;i<kernel_analysis->candidates_cnt;i++)
	{		
		printf("\n\t %s \t %d \t %d ",analysis_tableu[i].array_info->name,analysis_tableu[i].array_info->is_in_texture ,analysis_tableu[i].array_info->is_in_surface);
	}
	printf("\n\n");
}

enum RWbar get_updated_status(enum RWbar old, bool is_write,bool is_in_loop)
{
	printf("\n Is in Loop %d",is_in_loop);
	switch(old) 
        {
	      case none :
		 if(is_write)
			return write;
		 else if(!is_in_loop)
			return read;
		 else
			return read_inside_loop;		
	      case invalid :
			return invalid;
	      case error :
		 	return error;
	      case write :
		 if(is_write)
			return write;
		 else
			return invalid;
	      case read :
		 if(is_write && !is_in_loop)
			return write;		
		 else if(!is_write &&!is_in_loop)
			return read; 
		else if(is_write && is_in_loop)
			return write;
		else if(!is_write && is_in_loop)
			return read_inside_loop;
	     case read_inside_loop :
		 if(is_write && is_in_loop)
			return invalid;
		 else if(!is_write && is_in_loop)
			return read_inside_loop;
		 else if(is_write && !is_in_loop)
			return write;
		 else if(!is_write && !is_in_loop)
			return read;	
	   }
	
}
int is_candidate(const char *array_name,struct candidate_info * analysis_tableu,int candidates_cnt)
{
	int i;
	for(i=0;i<candidates_cnt;i++)
	{
		if(!strcmp(array_name,analysis_tableu[i].array_info->name))
		{
			return i;	
		}
	}
	
	return -1;
}

void update_analysis(struct candidate_info * analysis_tableu, int array_index, int kernel_index, bool is_access_write, int is_in_loop)
{
	if(is_in_loop<0)
	{
		// throw error; 
	}
	printf("\n\n [update] %s %d %d",analysis_tableu[array_index].array_info->name, is_access_write, is_in_loop);
enum RWbar updated = get_updated_status(analysis_tableu[array_index].latest_access[kernel_index],is_access_write,is_in_loop==0?false:true);
		printf("\n [update] milale %d \n",updated);
		printf(" Access is %d %d",is_access_write,is_in_loop);
		fflush(stdout);
	analysis_tableu[array_index].latest_access[kernel_index] = updated;
}

int for_each_array_access(__isl_keep pet_expr *expr, void *user)
{
	struct pet_traversal_arg * arg = (struct pet_traversal_arg *) user;
	struct kernel_analysis * kernel_analysis;
	struct candidate_info * analysis_tableu;
	isl_ast_expr *ast_expr;
	int is_access;
	isl_id_to_ast_expr *ref2expr;
	ref2expr = (isl_id_to_ast_expr *) arg->ref2expr;
	kernel_analysis = (struct kernel_analysis *) arg-> kernel_analysis;

	ast_expr = isl_id_to_ast_expr_get(ref2expr,pet_expr_access_get_ref_id(expr));
	is_access = isl_ast_expr_get_type(ast_expr) == isl_ast_expr_op &&
		isl_ast_expr_get_op_type(ast_expr) == isl_ast_op_access;

	

	const char * array_name = get_name(ast_expr);
	
	if(array_name!=NULL)
	{
		analysis_tableu = kernel_analysis->analysis_tableu;
		int kernel_index = kernel_analysis->visited_kernel_cnt;
		int array_index = is_candidate(array_name,analysis_tableu,kernel_analysis->candidates_cnt);
		if(array_index>=0)
		{
			bool is_access_write = (pet_expr_access_is_write(expr)==1)?true:false;
			analysis_tableu[array_index].array_info->is_write_only = (analysis_tableu[array_index].array_info->is_write_only &  is_access_write);
			
			if(kernel_analysis->is_annotating)
			{
				printf("\n\n annotating started");
				fflush(stdout);
				struct surf_read_expressions * surf_read_list = kernel_analysis->read_surface_to_temp;
				int temp_cnt = kernel_analysis->temp_cnt;
			temp_cnt = annotate_node(expr,ref2expr,&analysis_tableu[array_index],is_access_write,&surf_read_list,temp_cnt);
				kernel_analysis->read_surface_to_temp = surf_read_list;
				kernel_analysis->temp_cnt=temp_cnt;
				printf(" \n\n ----- annotating done !");			
			}
			else
			{
		
				if(is_access_write && arg->simulate_R_W)
				{
					// simulate read access for +=, -= *=, and /= 
					printf("\n\n Updating analysis");
					fflush(stdout);
					update_analysis (analysis_tableu,array_index,kernel_index, false,kernel_analysis->kernel_loop_depth_count);

				}

		printf("\n\n Updating analysis");
		fflush(stdout);
		update_analysis(analysis_tableu,array_index,kernel_index, is_access_write,kernel_analysis->kernel_loop_depth_count);
			}
		}
	}
	isl_ast_expr_free(ast_expr);
	return 0;
}

void traverse_pet_tree(__isl_keep pet_tree *tree,isl_id_to_ast_expr *ref2expr,void * user)
{

	struct pet_traversal_arg * arg = (struct pet_traversal_arg *)malloc(sizeof(struct pet_traversal_arg));
	arg->kernel_analysis = (struct kernel_analysis *)user;
	arg->ref2expr = ref2expr;


	arg->simulate_R_W = false;
	if(!arg->kernel_analysis->is_annotating)
	{	
		enum pet_op_type op_type = pet_expr_op_get_type(tree->u.e.expr);
		if(op_type == pet_op_add_assign || op_type == pet_op_sub_assign || op_type == pet_op_mul_assign || op_type == pet_op_div_assign)
		{
			printf("\n simulating R_W ");
			arg->simulate_R_W = true;
		}
	}	
	//if(arg->kernel_analysis->is_annotating)
	//{
	//	pet_tree_foreach_access_expr(tree,&for_each_array_access_mark_texture,arg);
	//}
	//else
	//{
		printf("checking here -------> %d ",arg->kernel_analysis->is_annotating);
		pet_tree_foreach_access_expr_postorder(tree,&for_each_array_access,arg);		
	//}	
}

isl_bool visit_kernel_stmts(__isl_keep isl_ast_node *node, void *user)
{
	isl_id * id;
	struct ppcg_kernel_stmt *stmt;
	//bool * is_update;
	id =isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);
	if(stmt==NULL)
	{
		//printf("Stmt NULL");
		return isl_bool_true;
	}
	if(stmt->type==ppcg_kernel_copy)
	{
		// some analysis here
		return isl_bool_true;
	}
	else if(stmt->type==ppcg_kernel_domain)
	{
		struct pet_stmt * pet_stmt = stmt->u.d.stmt->stmt;
		isl_id_to_ast_expr *ref2expr = stmt->u.d.ref2expr;
		pet_tree *tree = pet_stmt->body;
		//is_update = (bool *)user;
		traverse_pet_tree(tree,ref2expr,user);
	}
	return isl_bool_true;
}

void mark_is_array_written(struct ppcg_kernel * kernel, struct kernel_analysis * kernel_analysis)
{
	int i,j;
	for(i=0;i<kernel->n_array;i++)
	{
		int array_index,kernel_index;
	array_index = is_candidate(kernel->array[i].array->name,kernel_analysis->analysis_tableu,kernel_analysis->candidates_cnt);
		if(array_index>=0)
		{
			kernel_index=kernel->sequence_no;
			if(kernel_analysis->analysis_tableu[array_index].latest_access[kernel_index]==write)
				kernel->array[i].is_written = true;
		}	
	}
}

void traverse_kernel(struct ppcg_kernel *kernel, struct kernel_analysis * kernel_analysis)
{
	isl_ast_node* kernel_tree = kernel->tree;
	kernel_analysis->read_surface_to_temp = NULL;
	if(!kernel_analysis->is_annotating)
		isl_ast_node_foreach_descendant_top_down_analysis(kernel_tree,&visit_kernel_stmts,(void*)kernel_analysis);
	else
		isl_ast_node_foreach_descendant_top_down(kernel_tree,&visit_kernel_stmts,(void*)kernel_analysis);		
	if(kernel_analysis->is_annotating)
	{
		/* TODO : THis might be redundent store home here */
		kernel->decl=kernel_analysis->read_surface_to_temp;
		kernel_analysis->read_surface_to_temp=NULL;
		mark_is_array_written(kernel,kernel_analysis);
	}
	return ;
}

isl_bool visit_gpu_node(__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	int is_user;
	struct ppcg_kernel *kernel;
	struct ppcg_kernel_stmt *stmt;
	struct kernel_analysis * kernel_analysis;

	if(isl_ast_node_get_type(node) != isl_ast_node_user)
		return isl_bool_true;
	

	id = isl_ast_node_get_annotation(node);
	if(!id)
	{
		return isl_bool_false;
	}
	is_user = !strcmp(isl_id_get_name(id), "user");
	kernel = is_user ? NULL : isl_id_get_user(id);
	stmt = is_user ? isl_id_get_user(id) : NULL;
	isl_id_free(id);

	if (is_user)
		printf("\n statement ! ");
	else
	{
		// Now no need : Initialization needed for RWbar
		 
		printf("\n new kernel !");
		fflush(stdout);
		kernel_analysis = (struct kernel_analysis *)user;
		if(kernel_analysis->is_annotating)
		{
			surf_write_temp_counter = 0; 
			kernel_analysis->temp_cnt = 0;
		}
		kernel_analysis->kernel_loop_depth_count=0;
		traverse_kernel(kernel,kernel_analysis);
		if(!kernel_analysis->is_annotating)
		{
			kernel->sequence_no=kernel_analysis->visited_kernel_cnt;
			kernel_analysis->visited_kernel_cnt++;
		}					
	}
	return isl_bool_true;	
}	

void traverse_gpu_code(isl_ast_node *tree,struct kernel_analysis * kernel_analysis_arg)
{
	isl_ast_node_foreach_descendant_top_down(tree,&visit_gpu_node,(void *)kernel_analysis_arg);
	return ;
}

void perform_static_analysis(isl_ast_node *tree,struct kernel_analysis * kernel_analysis_arg)
{
	kernel_analysis_arg->is_annotating = false;
	return traverse_gpu_code(tree,kernel_analysis_arg);

}	
void annotate(isl_ast_node *tree,struct kernel_analysis * kernel_analysis)
{
	kernel_analysis->is_annotating = true;
	traverse_gpu_code(tree,kernel_analysis);
}

void setup_candidate_info(struct candidate_info * table, int index ,struct gpu_array_info * array, int no_of_kernels)
{
	int j;
	table[index].array_info = array;
	table[index].latest_access = malloc(sizeof(enum RWbar)*no_of_kernels);
	for(j =0; j< no_of_kernels ; j++)
		table[index].latest_access[j] = none;	
	table[index].array_info->is_write_only = true;
}

struct kernel_analysis * setup_candidate_arrays(struct gpu_prog *prog)
{
	int i,j;
	if(prog->scop!=NULL)
	{
		struct texture_candidate **candidate_list = NULL;
		struct kernel_analysis * kernel_analysis_arg;		
		int no_of_candidates=0;
		for(i=0;i<prog->n_array;i++)
		{
			if(prog->array[i].n_index>0)
			{
				printf("\n %s",prog->array[i].name);
				no_of_candidates++;	
			}
		}
		printf(" \n Candidates are : %d",no_of_candidates);
		if(no_of_candidates==0)
			return NULL;
		int no_of_kernels = prog->scop->kernels_extracted;
		struct candidate_info * analysis_table;
		printf("\n Setting up data structures");
		printf(" \n #Kernels extracted are : %d",no_of_kernels);
		printf(" \n Candidates are : %d",no_of_candidates);

	 	analysis_table = (struct candidate_info *)malloc(sizeof(struct candidate_info)*no_of_candidates);
		int j=0;
		for(i=0;i<prog->n_array;i++)
		{
			if(prog->array[i].n_index)
			{
				printf("\n\t\t %s ",prog->array[i].name);
				setup_candidate_info(analysis_table,j,&prog->array[i],no_of_kernels);
				j++;
			}		
		}
		fflush(stdout);	
		kernel_analysis_arg = (struct kernel_analysis *)malloc(sizeof(struct kernel_analysis));
		kernel_analysis_arg-> analysis_tableu = analysis_table; 
		kernel_analysis_arg-> visited_kernel_cnt = 0;
		kernel_analysis_arg-> candidates_cnt = no_of_candidates;
		kernel_analysis_arg-> kernel_cnt = no_of_kernels;
		kernel_analysis_arg-> kernel_loop_depth_count = 0;
		return kernel_analysis_arg;
	}
	else
		return NULL ;
}



void texture_optimization(struct gpu_prog *prog,isl_ast_node *tree)
{
	// FIXME if double then not a candidate
	is_ccp_greater_than_two = (prog->scop->options->use_surface_memory)? true : false;
	is_ccp_greater_than_two? printf("\n Using surfaces \n") : printf("\n Not Using surfaces \n");
	bool is_memory_opts = (prog->scop->options->memory_opts)? true : false;	
	if(!is_memory_opts)
		return;
	printf("Here");

	struct kernel_analysis * kernel_analysis_arg =  setup_candidate_arrays(prog);	
	if(kernel_analysis_arg==NULL)
		return ;	
	if(tree==NULL)
		return ;
	perform_static_analysis(tree,kernel_analysis_arg);	

	print_analysis(kernel_analysis_arg);

	decide_surface_or_texture(kernel_analysis_arg);

	decide_texture_profitability(kernel_analysis_arg);
	
	annotate(tree,kernel_analysis_arg);
	//free_variables();
}

