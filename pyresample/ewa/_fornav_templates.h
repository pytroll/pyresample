#ifndef _FORNAV_TEMPLATES_H
#define _FORNAV_TEMPLATES_H

// The types used for all of the intermediate storage between input swath pixels and output grid pixels
// Mainly here for testing, but should never be a non-floating type
typedef float weight_type;
typedef float ewa_param_type;
typedef float accum_type;
//typedef double weight_type;
//typedef double ewa_param_type;
//typedef double accum_type;

//const weight_type EPSILON = 1e-8;
#define EPSILON (1e-8)

typedef struct {
    ewa_param_type a;
    ewa_param_type b;
    ewa_param_type c;
    ewa_param_type f;
    ewa_param_type u_del;
    ewa_param_type v_del;
} ewa_parameters;

typedef struct {
    int count;
    weight_type min;
    weight_type distance_max;
    weight_type delta_max;
    weight_type sum_min;
    weight_type alpha;
    weight_type qmax;
    weight_type qfactor;
    weight_type *wtab;
} ewa_weight;

int initialize_weight(size_t chan_count, unsigned int weight_count, weight_type weight_min, weight_type weight_distance_max,
        weight_type weight_delta_max, weight_type weight_sum_min, ewa_weight *ewaw);

void deinitialize_weight(ewa_weight *ewaw);

accum_type **initialize_grid_accums(size_t chan_count, size_t grid_cols, size_t grid_rows);

weight_type **initialize_grid_weights(size_t chan_count, size_t grid_cols, size_t grid_rows);
void deinitialize_grids(size_t chan_count, void **grids);

template<typename CR_TYPE>
int compute_ewa_parameters(size_t swath_cols, size_t swath_rows, CR_TYPE *uimg, CR_TYPE *vimg, ewa_weight *ewaw, ewa_parameters *ewap);

template<typename CR_TYPE, typename IMAGE_TYPE>
extern int compute_ewa(size_t chan_count, int maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        CR_TYPE *uimg, CR_TYPE *vimg,
        IMAGE_TYPE **images, IMAGE_TYPE img_fill, accum_type **grid_accums, weight_type **grid_weights,
        ewa_weight *ewaw, ewa_parameters *ewap);

template<typename CR_TYPE, typename IMAGE_TYPE>
extern int compute_ewa_single(int maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        CR_TYPE *uimg, CR_TYPE *vimg,
        IMAGE_TYPE *image, IMAGE_TYPE img_fill, accum_type *grid_accum, weight_type *grid_weight,
        ewa_weight *ewaw, ewa_parameters *ewap);

template<typename GRID_TYPE> unsigned int write_grid_image(GRID_TYPE *output_image, GRID_TYPE fill,
        size_t grid_cols, size_t grid_rows,
        accum_type *grid_accum, weight_type *grid_weights,
        int maximum_weight_mode, weight_type weight_sum_min);

#endif
