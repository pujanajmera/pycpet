# include <stdio.h>
# include <stdlib.h>
# include <stdbool.h>
# include <omp.h>
# include <math.h>
# include <string.h>
//#include <gsl/gsl_vector.h>
//#include <gsl/gsl_blas.h>
//# include <mpi.h>

// faster: gcc -fopenmp math_module.c -o math_module.so -shared -fPIC -O3 -march=native -funroll-loops -ffast-math
// compile without mpi w gcc -fopenmp math_module.c -o math_module.so -shared -fPIC
// compile w mpicc -fPIC -shared -o matmulmodule.so matmulmodule.c -fopenmp

void sparse_dot(double* ret, int* indptr, int indptrlen, int* indA, int lenindA, double* A, int lenA, double* B, int size_array){
    int rows = indptrlen - 1;
    int i; 
    int j;
    int provided = 0;
    int ptr;
    int ptrtemp;
    double sum;
    int numInRow;
    ptr = 0;
    
    // MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    #pragma omp parallel shared(A, B, ret) private(i,j)
    {
        for(i=1; i < indptrlen; i++){
            sum = 0;
            numInRow= indptr[i] - indptr[i-1];
            ptrtemp = ptr;
            ptr += numInRow;


            for(j = ptrtemp; j < ptr; j++){
                    sum += A[j]*B[indA[j]];

            }
        
            ret[i-1] = sum; 

        }
    }
    // MPI_Finalize();
}


void dot(double* ret, double* A, double* B, int rows, int cols){
    int i; 
    int j;
    int provided = 0;
    double sum;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    //#pragma omp parallel shared(A, B, ret) private(i,j)
    {
        for(i=0; i < rows; i++){
            sum = 0;
            for(j = 0; j < cols; j++){
                    sum += A[i*cols + j]*B[j];

            }
            ret[i] = sum; 
        }
    }
    
    //MPI_Finalize();
}


void cross_product(float *u, float *v, float *product)
{
        float p1 = u[1]*v[2]
                - u[2]*v[1];

        float p2 = u[2]*v[0]
                - u[0]*v[2];

        float p3 = u[0]*v[1]
                - u[1]*v[0];

        product[0] = p1;
        product[1] = p2;
        product[2] = p3;

}


void norm(float *u, float *norm_val)
{
    *norm_val = sqrt(
        pow(u[0], 2) + 
        pow(u[1], 2) + 
        pow(u[2], 2)
    );
}


double euclidean_dist(float* x_0, float* x_1){
    float sum = 0;
    for(int i = 0; i < 3; i++){
        sum += pow(fabsf(x_0[i] - x_1[i]), 2);
    }
    return sqrt(sum);
}


double curve(float* x_0, float* x_1){
    // compute cross of x_0 and x_1
    float cross_prod[3];
    float norm_cross_prod;
    float norm_denom;

    cross_product(x_0, x_1, cross_prod);
    norm(cross_prod, &norm_cross_prod);
    norm(x_0, &norm_denom);
    float curve_ret = norm_cross_prod / pow(norm_denom, 3);

    return curve_ret;

}


void vecaddn(float* ret, float* A, float* B, int lenA){
    int i; 
    //#pragma omp parallel shared(A, B, ret) private(i)
    {
    for (i = 0; i < lenA; i++) {
                ret[i] = A[i] + B[i];
            }
    }   
}


void einsum_operation_batch(int batch, int rows, float r_mag[batch][rows], float Q[rows], float R[batch][rows][3], float result[batch][3]){
    int i; 
    int j;
    int k;
    int provided = 0;
    float factor = 14.3996451;
    float r_0;
    float r_1;
    float r_2;
    float ele_1;
    float ele_2;
    float ele_3;
    float compute_singular;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    //#pragma omp parallel shared(R, r_mag, Q, result) private(i,j,k)
    {
        for(k=0; k < batch; k++){
            float sum[3] = {0.0, 0.0, 0.0};
            for(i=0; i < rows; i++){
                r_0 = R[k][i][0];
                r_1 = R[k][i][1];
                r_2 = R[k][i][2];
                ele_1 = r_mag[k][i];
                ele_2 = Q[i];
                ele_3 = r_mag[k][i];
                compute_singular = factor * ele_2 * ele_3;
                sum[0] += compute_singular * r_0;
                sum[1] += compute_singular * r_1;
                sum[2] += compute_singular * r_2;

            }
            result[k][0] = sum[0];
            result[k][1] = sum[1];
            result[k][2] = sum[2];
        }
    }
    //MPI_Finalize();
}


void einsum_operation(int rows, float r_mag[rows], float Q[rows], float R[rows][3], float result[3]){
    int i; 
    int j;
    int provided = 0;
    float sum[3] = {0.0, 0.0, 0.0};
    float factor = 14.3996451;
    float r_0;
    float r_1;
    float r_2;
    float ele_1;
    float ele_2;
    float ele_3;
    float compute_singular;
    // compute "ij,ij,ij->j" einsum
    {
        for(i=0; i < rows; i++){
            
            r_0 = R[i][0];
            r_1 = R[i][1];
            r_2 = R[i][2];
            ele_2 = Q[i];
            ele_3 = r_mag[i]; 
            
            compute_singular = factor * ele_2 * ele_3;
            sum[0] += compute_singular * r_0;
            sum[1] += compute_singular * r_1;
            sum[2] += compute_singular * r_2;

        }
        result[0] = sum[0];
        result[1] = sum[1];
        result[2] = sum[2];
    }
    //MPI_Finalize();
}


void einsum_ij_i_batch(int batch, int rows, int cols, float A[batch][rows][cols], float ret[batch][rows]){
    int i; 
    int j;
    int k;
    int provided = 0;
    float sum;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize 
    //#pragma omp parallel shared(A, ret) private(i,j,k)
    {
        for(k=0; k < batch; k++){
            for(i=0; i < rows; i++){
                sum = 0;
                for(j = 0; j < cols; j+=1){
                        sum += A[k][i][j];
                }
                ret[k][i] = sum;    
            }
        }
    }
    //MPI_Finalize();
}


void einsum_ij_i(int rows, int cols, float A[rows][cols], float ret[rows]){
    int i; 
    int j;
    int provided = 0;
    float sum;
    //MPI_Init_thread(NULL,NULL,  MPI_THREAD_MULTIPLE, &provided); // Initialize
    //#pragma omp parallel shared(A, ret) private(i,j)
    {
    
        for(i=0; i < rows; i++){
            sum = 0;
            for(j = 0; j < cols; j+=1){
                    sum += A[i][j];
            }
            ret[i] = sum;    
        }
    }
}


void calc_field(float E[3], float x_init[3], int n_charges, float x[n_charges][3], float Q[n_charges]){
    // calculate the field

    // subtract x_init from x
    float R[n_charges][3];
    float r_mag[n_charges];
    float E_array[3];
    float r_sq[n_charges][3];
    float r_mag_sq[n_charges];
    
    // make R a 2D array
    for (int j = 0; j < 3; j++)
    {
        float x_init_single = x_init[j];
        //# pragma omp parallel for
        for (int i = 0; i < n_charges; i++)
        {
            R[i][j] = x_init_single - x[i][j];
            r_sq[i][j] = pow(R[i][j], 2);
        }
    }
    
    einsum_ij_i(n_charges, 3, r_sq, r_mag_sq); // r_mag_sq might be wrong shape here
    
    //# pragma omp parallel for
    for (int i = 0; i < n_charges; i++)
    {
        // elementwise raise to -3/2
        r_mag[i] = pow(r_mag_sq[i], -1.5); // if this is zero then we have a problem
    }
    
    // compute einsum operation
    einsum_operation(n_charges, r_mag, Q, R, E_array); //

    E[0] = E_array[0] ;
    E[1] = E_array[1] ;
    E[2] = E_array[2] ;

}


void calc_field_base(float E[3], float x_init[3], int n_charges, float x[n_charges][3], float Q[n_charges]){
    // calculate the field

    // subtract x_init from x
    float R[3];
    float r_mag[n_charges];
    //float r_sq[n_charges][3];
    //float r_mag_sq[n_charges];
    float r_mag_cube;
    float factor = 14.3996451;
    float r_norm;
    float E_temp[n_charges][3];

    //# pragma omp parallel for shared(E, x_init, n_charges, x, Q) private(r_mag_cube, r_norm, R)
    # pragma omp parallel for
    for (int i = 0; i < n_charges; i++)
    {
        //get difference of x and x_init
        R[0] = x_init[0] - x[i][0];
        R[1] = x_init[1] - x[i][1];
        R[2] = x_init[2] - x[i][2]; 

        r_norm = sqrt(pow(R[0], 2) + pow(R[1], 2) + pow(R[2], 2));
        r_mag_cube = pow(r_norm, -3);
        //r_mag[i] = pow(r_norm, -3);

        E_temp[i][0] = factor * r_mag_cube * Q[i] * R[0];
        E_temp[i][1] = factor * r_mag_cube * Q[i] * R[1];
        E_temp[i][2] = factor * r_mag_cube * Q[i] * R[2];
    }

    for (int i = 0; i < n_charges; i++)
    {
        E[0] += E_temp[i][0];
        E[1] += E_temp[i][1];
        E[2] += E_temp[i][2];
    }
}

// Compute electric field by giving batches of 100
void compute_batched_field(int total_points, int batch_size, int n_charges, float x_0[total_points][3], float x[n_charges][3], float Q[n_charges], float E[total_points][3]) {
    for(int start = 0; start < total_points; start += batch_size) {
        int current_batch_size = batch_size;
        if (start + current_batch_size > total_points)
            current_batch_size = total_points - start;

        float (*R)[n_charges][3] = malloc(current_batch_size * sizeof(*R));
        float (*r_mag)[n_charges] = malloc(current_batch_size * sizeof(*r_mag));
        float (*E_batch)[3] = malloc(current_batch_size * sizeof(*E_batch));

        for(int k = 0; k < current_batch_size; k++){
            for(int i = 0; i < n_charges; i++){
                float sum_sq = 0.0;
                for(int d = 0; d < 3; d++){
                    float diff = x_0[start + k][d] - x[i][d];
                    R[k][i][d] = diff;
                    sum_sq += diff * diff;
                }
                r_mag[k][i] = 1.0f / powf(sqrtf(sum_sq), 3);
            }
        }

        einsum_operation_batch(current_batch_size, n_charges, r_mag, Q, R, E_batch);
        memcpy(&E[start], E_batch, current_batch_size * sizeof(*E_batch));

        free(R);
        free(r_mag);
        free(E_batch);
    }
}

void compute_looped_field(int total_points, int n_charges, float x_0[total_points][3], float x[n_charges][3], float Q[n_charges], float E[total_points][3]) {
    
    float epsilon = 1e-6f;  // Softening factor to avoid singularity
    
    float R[n_charges][3];
    float r_mag[n_charges];

    float factor = 14.3996451;
    
    for (int start = 0; start < total_points; start++) {
        
        float E_temp[3] = {0.0f, 0.0f, 0.0f};
        
        // Compute displacement vectors R = x_0[start] - x[i]
        for (int i = 0; i < n_charges; i++) {
            float dx = x_0[start][0] - x[i][0];
            float dy = x_0[start][1] - x[i][1];
            float dz = x_0[start][2] - x[i][2];
            
            R[i][0] = dx;
            R[i][1] = dy;
            R[i][2] = dz;
            
            r_mag[i] = dx*dx + dy*dy + dz*dz;  // Squared distance
        }
        
        // Compute inverse r^3 safely
        for (int i = 0; i < n_charges; i++) {
            float safe_r2 = fmaxf(r_mag[i], epsilon);  // Avoid r=0
            float rinv = 1.0f / sqrtf(safe_r2);
            r_mag[i] = rinv * rinv * rinv;  // Equivalent to r^-3
        }
        
        // Compute Electric Field Sum
        for (int i = 0; i < n_charges; i++) {
            float scale = factor * Q[i] * r_mag[i];
            E_temp[0] += scale * R[i][0];
            E_temp[1] += scale * R[i][1];
            E_temp[2] += scale * R[i][2];
        }
        
        // Store result
        E[start][0] = E_temp[0];
        E[start][1] = E_temp[1];
        E[start][2] = E_temp[2];
    }
}

void calc_esp_base(float ESP[1], float x_init[3], int n_charges, float x[n_charges][3], float Q[n_charges]){
    // calculate the electrostatic potential

    // subtract x_init from x
    float R[3];
    float r_mag[n_charges];
    //float r_sq[n_charges][3];
    //float r_mag_sq[n_charges];
    float r_mag_inv;
    float factor = 14.3996451;
    float r_norm;
    float ESP_temp[n_charges][3];

    //# pragma omp parallel for shared(E, x_init, n_charges, x, Q) private(r_mag_cube, r_norm, R)
    # pragma omp parallel for
    for (int i = 0; i < n_charges; i++)
    {
        //get difference of x and x_init
        R[0] = x_init[0] - x[i][0];
        R[1] = x_init[1] - x[i][1];
        R[2] = x_init[2] - x[i][2]; 

        r_norm = sqrt(pow(R[0], 2) + pow(R[1], 2) + pow(R[2], 2));
        r_mag_inv = pow(r_norm, -1);
        //r_mag[i] = pow(r_norm, -3);

        ESP_temp[i][0] = factor * r_mag_inv * Q[i];
    }

    for (int i = 0; i < n_charges; i++)
    {
        ESP[0] += ESP_temp[i][0];
    }
}


void propagate_topo(float result[3], float x_init[3], int n_charges, float x[n_charges][3], float Q[n_charges], float step_size){
    // propagate the topology
    float E[3] = {0.0, 0.0, 0.0};
    float E_norm;
    //printf("propagating!!");

    //calc_field(E, x_init, n_charges, x, Q);
    calc_field_base(E, x_init, n_charges, x, Q);
    
    norm(E, &E_norm);
    for (int i = 0; i < 3; i++)
    {
        result[i] = x_init[i] + step_size * E[i] / (E_norm);
    }
}


void thread_operation(int n_charges, int n_iter, float step_size, float x_0[3], float dimensions[3], float x[n_charges][3], float Q[n_charges], float ret[2]){
    bool bool_inside = true;
    float x_init[3] = {x_0[0], x_0[1], x_0[2]};
    float x_overwrite[3] = {x_0[0], x_0[1], x_0[2]};
    // print x_init
    // printf("x_init %f %f %f\n", x_init[0], x_init[1], x_init[2]);
    int i; 
    float half_length = dimensions[0];
    float half_width = dimensions[1];
    float half_height = dimensions[2];
    
    for (i = 0; i < n_iter; i++) {

        propagate_topo(x_overwrite, x_overwrite, n_charges, x, Q, step_size);
        // overwrite x_init with x_overwrite

        if (
            x_overwrite[0] < -half_length || 
            x_overwrite[0] > half_length || 
            x_overwrite[1] < -half_width || 
            x_overwrite[1] > half_width || 
            x_overwrite[2] < -half_height || 
            x_overwrite[2] > half_height){
            bool_inside = false;
        }

        // printf("%f %f %f\n", x_overwrite[0], x_overwrite[1], x_overwrite[2]);

        if (!bool_inside){
            // printf("Breaking out of loop at iteration: %i\n", i);
            break;
        }
        // printf("x_final @ iter %i out of %i %f %f %f\n", i, n_iter, x_overwrite[0], x_overwrite[1], x_overwrite[2]);
        
    }
        
    
    float x_init_plus[3];
    float x_init_plus_plus[3];
    float x_0_plus[3];
    float x_0_plus_plus[3];    

    propagate_topo(x_init_plus, x_init, n_charges, x, Q, step_size);
    propagate_topo(x_init_plus_plus, x_init_plus, n_charges, x, Q, step_size);
    propagate_topo(x_0_plus, x_overwrite, n_charges, x, Q, step_size);
    propagate_topo(x_0_plus_plus, x_0_plus, n_charges, x, Q, step_size);

    float curve_arg_1[3];
    float curve_arg_2[3];
    float curve_arg_3[3];
    float curve_arg_4[3];
    
    for (int i = 0; i < 3; i++) {
        curve_arg_1[i] = x_init_plus[i] - x_init[i];
        curve_arg_2[i] = x_init_plus_plus[i] - 2* x_init_plus[i] + x_init[i];
        curve_arg_3[i] = x_0_plus[i] - x_overwrite[i];
        curve_arg_4[i] = x_0_plus_plus[i] - 2* x_0_plus[i] + x_overwrite[i];
    }

    float curve_init = curve(curve_arg_1, curve_arg_2);
    float curve_final = curve(curve_arg_3, curve_arg_4);
    float curve_mean = (curve_init + curve_final) / 2;
    float dist = euclidean_dist(x_0, x_overwrite);
    

    ret[0] = dist;
    ret[1] = curve_mean;

}


