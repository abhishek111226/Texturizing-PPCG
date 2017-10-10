/*
 * Copyright 2016
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Abhishek Patwardhan
 * IIT Hyderabad, India
 *	
 */
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include "gpu.h"
#include "gpu_print.h"
#include "pet/expr.h"
struct gpu_prog;
enum RWbar 
{
	write,
	read,
	invalid,
	error, 
	none,
	read_inside_loop
}; 
enum data_transfer
{
	to_device,
	from_device,
	among_device	
};
struct candidate_info
{
	struct gpu_array_info* array_info;
	enum RWbar * latest_access;
};

struct kernel_analysis
{
	struct candidate_info * analysis_tableu;
	int visited_kernel_cnt;
	int candidates_cnt;
	int kernel_cnt;
	bool is_annotating;
	struct surf_read_expressions * read_surface_to_temp;
	int temp_cnt;
	int kernel_loop_depth_count;
};

struct pet_traversal_arg
{
	struct kernel_analysis * kernel_analysis;
	isl_id_to_ast_expr *ref2expr;
	bool simulate_R_W;
};

struct surf_read_expressions
{
	char * temp_name;
	char * type;
	isl_ast_expr * access_expr;
	isl_ast_expr * surface_read_expr;
	struct surf_read_expressions * next;
};
static surf_write_temp_counter=0; 

enum tex_or_surf in_texture_or_surface(const char *, char ** type);
__isl_give isl_printer *bind_device_textures(__isl_take isl_printer *, struct gpu_prog *);
__isl_give isl_printer *declare_device_textures(__isl_take isl_printer *, struct gpu_prog *);
__isl_give isl_printer * texture_array_access(__isl_take isl_printer *, __isl_keep isl_ast_expr *);
static __isl_give isl_printer *if_texture_print_access(__isl_take isl_printer *,__isl_keep isl_ast_expr *, bool);
void insert_surf_read_access(__isl_keep isl_ast_expr * expr, struct surf_read_expressions ** read_surf_list,__isl_keep isl_ast_expr * expr_old, char * type, char * temp_name);
void alter_access(__isl_keep isl_ast_expr *expr, char * name);
bool is_ccp_greater_than_two;
const char* get_name(isl_ast_expr *ast_expr);
void clear_texture_flags(struct gpu_prog * prog);
 __isl_give isl_printer * print_texture_access(__isl_take isl_printer *p,isl_ast_expr *expr);
__isl_give isl_printer * print_surface_write_call(__isl_take isl_printer *p, int temp_id, char * type,int surf_dim,__isl_keep isl_ast_expr *expr_destination);
void gentemp( char ** temp, isl_ast_expr * expr, int surf_temp_cnt);
void surface_temp_decide_types(int n_array,struct gpu_array_info * arrays);
void insert_tex_decl(struct gpu_array_info*  arr);
 __isl_give isl_printer * print_surface_read_access(__isl_take isl_printer *p,isl_ast_expr *expr, char *dest, char * type);
const char * extract_array_name_from_access(__isl_keep isl_ast_expr *expr);
__isl_give isl_ast_expr *isl_ast_expr_dup(__isl_keep isl_ast_expr *expr);
__isl_give isl_printer *print_surface_read_to_temp(__isl_take isl_printer *p, struct ppcg_kernel *kernel);
__isl_give isl_ast_expr* construct_surface_read_access(__isl_keep isl_ast_expr * array_access,char * type, char *temp);
__isl_give isl_printer *print_surface_reads(__isl_take isl_printer *p, struct surf_read_expressions * surface_reads);
void isl_ast_expr_modify_access_to_tex_call(__isl_keep isl_ast_expr * expr,bool is_linearize);
void remove_loop_read_marks(struct kernel_analysis * kernel_analysis);
__isl_give isl_map * create_transformed_access(isl_multi_pw_aff *mpa);
const char * get_array_name(isl_map * taccess);
void analyze_cost_spatial_locality(isl_map *taccess, const char * array_name);
isl_stat isl_ast_node_foreach_descendant_top_down_analysis(
	__isl_keep isl_ast_node *node,
	isl_bool (*fn)(__isl_keep isl_ast_node *node, void *user), void *user);
