#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <math.h>
//% setenv LD_LIBRARY_PATH /usr/local/lib
void cross_product(const gsl_vector *u, const gsl_vector *v, gsl_vector *product)
{
        double p1 = gsl_vector_get(u, 1)*gsl_vector_get(v, 2)
                - gsl_vector_get(u, 2)*gsl_vector_get(v, 1);

        double p2 = gsl_vector_get(u, 2)*gsl_vector_get(v, 0)
                - gsl_vector_get(u, 0)*gsl_vector_get(v, 2);

        double p3 = gsl_vector_get(u, 0)*gsl_vector_get(v, 1)
                - gsl_vector_get(u, 1)*gsl_vector_get(v, 0);

        gsl_vector_set(product, 0, p1);
        gsl_vector_set(product, 1, p2);
        gsl_vector_set(product, 2, p3);
}

void norm(const gsl_vector *u, double *norm_val)
{
    *norm_val = sqrt(
        pow(gsl_vector_get(u, 0), 2) + 
        pow(gsl_vector_get(u, 1), 2) + 
        pow(gsl_vector_get(u, 2), 2)
    );
}

double curve(gsl_vector* x_0, gsl_vector* x_1){
    // compute cross of x_0 and x_1
    gsl_vector* cross_prod = gsl_vector_alloc(3);
    double norm_cross_prod;
    double norm_denom;

    cross_product(x_0, x_1, cross_prod);
    norm(cross_prod, &norm_cross_prod);
    norm(x_0, &norm_denom);
    double curve_ret = norm_cross_prod / pow(norm_denom, 3);
    return curve_ret;

}

double euclidean_dist(gsl_vector* x_0, gsl_vector* x_1){
    double sum = 0;
    for(int i = 0; i < 3; i++){
        sum += pow(gsl_vector_get(x_0, i) - gsl_vector_get(x_1, i), 2);
    }
    return sqrt(sum);
}

float
main (void)
{
  int i;
  gsl_vector * v = gsl_vector_alloc (3);
  gsl_vector * x_init = gsl_vector_alloc (3);

  for (i = 0; i < 3; i++)
    {
      gsl_vector_set (v, i, 1.23 + i);
      gsl_vector_set (x_init, i, (1.23 + i) + 1);
    }

    
    double curve_val = curve(v, x_init);
    printf("curve_val: %f\n", curve_val);
  
}


