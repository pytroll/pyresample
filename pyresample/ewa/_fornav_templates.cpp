#include "Python.h"
#include <stddef.h>
#include "math.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "_fornav_templates.h"

template<typename IMAGE_TYPE> int __isnan(IMAGE_TYPE x) {
    // Return numpy's isnan for normal float arguments (see __isnan below for ints)
    return npy_isnan(x);
}

int __isnan(npy_int8 x) {
    // Sometimes input data may be integers, need to be able to handle those similarly
    return 0;
}

int initialize_weight(size_t chan_count, unsigned int weight_count, weight_type weight_min, weight_type weight_distance_max,
        weight_type weight_delta_max, weight_type weight_sum_min, ewa_weight *ewaw) {
  unsigned int idx;
  weight_type *wptr;

  ewaw->wtab = (weight_type *)calloc(weight_count, sizeof(weight_type));
  if (!ewaw->wtab) {
    return -1;
  }

  ewaw->count = weight_count;
  ewaw->min = weight_min;
  ewaw->distance_max = weight_distance_max;
  ewaw->delta_max = weight_delta_max;
  ewaw->sum_min = weight_sum_min;

  if (weight_count < 2) {
    // must be at least 2
    return -1;
  }
  if (weight_min <= 0.0) {
    // must be greater than 0
    return -1;
  }
  if (weight_distance_max <= 0.0) {
    // must be greater than 0
    return -1;
  }

  ewaw->qmax = ewaw->distance_max * ewaw->distance_max;
  ewaw->alpha = -log(ewaw->min) / ewaw->qmax;
  wptr = ewaw->wtab;
  for (idx=0; idx < weight_count; idx++) {
    wptr[idx] = exp(-ewaw->alpha * ewaw->qmax * idx / (ewaw->count - 1));
  }

  ewaw->qfactor = ewaw->count / ewaw->qmax;
  return 0;
}

void deinitialize_weight(ewa_weight *ewaw) {
  if (ewaw->wtab) {
    free(ewaw->wtab);
  }
}

accum_type **initialize_grid_accums(size_t chan_count, size_t grid_cols, size_t grid_rows) {
  accum_type **grid_accums = (accum_type **)malloc(chan_count * sizeof(accum_type *));
  unsigned int i;

  if (!grid_accums) {
    return NULL;
  }
  for (i=0; i < chan_count; i++) {
    grid_accums[i] = (accum_type *)calloc(grid_cols * grid_rows, sizeof(accum_type));
    if (!grid_accums[i]) {
      return NULL;
    }
  }

  return grid_accums;
}

weight_type **initialize_grid_weights(size_t chan_count, size_t grid_cols, size_t grid_rows) {
  weight_type **grid_weights = (weight_type **)malloc(chan_count * sizeof(weight_type *));
  unsigned int i;

  if (!grid_weights) {
    return NULL;
  }
  for (i=0; i<chan_count; i++) {
    grid_weights[i] = (weight_type *)calloc(grid_cols * grid_rows, sizeof(weight_type));
    if (!grid_weights[i]) {
      return NULL;
    }
  }

  return grid_weights;
}

void deinitialize_grids(size_t chan_count, void **grids) {
  unsigned int i;
  for (i = 0; i < chan_count; i++) {
    if (grids[i]) {
      free(grids[i]);
    }
  }
  free(grids);
}

template <typename CR_TYPE>
int compute_ewa_parameters(size_t swath_cols, size_t swath_rows, CR_TYPE *uimg, CR_TYPE *vimg, ewa_weight *ewaw, ewa_parameters *ewap) {
  ewa_param_type ux;
  ewa_param_type uy;
  ewa_param_type vx;
  ewa_param_type vy;
  ewa_param_type f_scale;

  // For testing: original C version uses doubles here
//  double ux;
//  double uy;
//  double vx;
//  double vy;
//  double f_scale;


  ewa_param_type d;
  ewa_param_type qmax;
  ewa_param_type distance_max;
  ewa_param_type delta_max;
  unsigned int rowsm1;
  unsigned int colsm1;
  unsigned int rowsov2;
  unsigned int col;
  ewa_parameters *this_ewap;

  qmax = ewaw->qmax;
  distance_max = ewaw->distance_max;
  delta_max = ewaw->delta_max;
  rowsm1 = swath_rows - 1;
  colsm1 = swath_cols - 1;
  rowsov2 = swath_rows / 2;

  for (col = 1, this_ewap=ewap + 1; col < colsm1; col++, this_ewap++) {
    ux = ((uimg[col - 1 + rowsov2 * swath_cols + 2] - uimg[col - 1 + rowsov2 * swath_cols]) / 2.0) * distance_max;
    vx = ((vimg[col - 1 + rowsov2 * swath_cols + 2] - vimg[col - 1 + rowsov2 * swath_cols]) / 2.0) * distance_max;
    uy = ((uimg[col + rowsm1 * swath_cols] - uimg[col]) / rowsm1) * distance_max;
    vy = ((vimg[col + rowsm1 * swath_cols] - vimg[col]) / rowsm1) * distance_max;

    // Handle geolocation being bad with a little bit of grace
    if (__isnan(ux) | __isnan(vx) | __isnan(uy) || __isnan(vy)) {
        this_ewap->a = 0;
        this_ewap->b = 0;
        this_ewap->c = 0;
        this_ewap->f = qmax;
        this_ewap->u_del = distance_max;
        this_ewap->v_del = distance_max;
        continue;
    }

    f_scale = ux * vy - uy * vx;
    f_scale = f_scale * f_scale;
    if (f_scale < EPSILON) {
      f_scale = EPSILON;
    }

    f_scale = qmax / f_scale;
    this_ewap->a = (vx * vx + vy * vy) * f_scale;
    this_ewap->b = -2.0 * (ux * vx + uy * vy) * f_scale;
    this_ewap->c = (ux * ux + uy * uy) * f_scale;

    d = 4.0 * this_ewap->a * this_ewap->c - this_ewap->b * this_ewap->b;
    if (d < EPSILON) {
      d = EPSILON;
    }
    d = ((4.0 * qmax) / d);
    this_ewap->f = qmax;
    this_ewap->u_del = sqrt(this_ewap->c * d);
    this_ewap->v_del = sqrt(this_ewap->a * d);

    if (this_ewap->u_del > delta_max) {
      this_ewap->u_del = delta_max;
    }
    if (this_ewap->v_del > delta_max) {
      this_ewap->v_del = delta_max;
    }
  }

  // Copy the parameters from the penultimate column to the last column (this_ewap should be at the final column)
  *this_ewap = *(this_ewap - 1);

  // Copy the parameters from the second column to the first column
  *ewap = *(ewap + 1);

  return 0;
}

template<typename CR_TYPE, typename IMAGE_TYPE>
int compute_ewa(size_t chan_count, int maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows, CR_TYPE *uimg, CR_TYPE *vimg,
        IMAGE_TYPE **images, IMAGE_TYPE img_fill, accum_type **grid_accums, weight_type **grid_weights, ewa_weight *ewaw, ewa_parameters *ewap) {
  // This was originally copied from a cython C file for 32-bit float inputs (that explains some of the weird parens and other syntax
  int got_point;
  unsigned int row;
  unsigned int col;
  ewa_parameters *this_ewap;
  int iu1;
  int iu2;
  int iv1;
  int iv2;
  int iu;
  int iv;
  CR_TYPE u0;
  CR_TYPE v0;
  weight_type ddq;
  weight_type dq;
  weight_type q;
  weight_type u;
  weight_type v;
  weight_type a2up1;
  weight_type au2;
  weight_type bu;
  weight_type weight;

  // Test: This is how the original fornav did its calculations
//  double u0;
//  double v0;
//  double ddq;
//  double dq;
//  double q;
//  double u;
//  double v;
//  double a2up1;
//  double au2;
//  double bu;
//  double weight;
//  double qfactor;

  int iw;
  IMAGE_TYPE this_val;
  unsigned int swath_offset;
  unsigned int grid_offset;
  size_t chan;

  got_point = 0;
  for (row = 0, swath_offset=0; row < swath_rows; row+=1) {
    for (col = 0, this_ewap = ewap; col < swath_cols; col++, this_ewap++, swath_offset++) {
      u0 = uimg[swath_offset];
      v0 = vimg[swath_offset];

      if (u0 < -this_ewap->u_del || v0 < -this_ewap->v_del || __isnan(u0) || __isnan(v0)) {
        continue;
      }

      iu1 = ((int)(u0 - this_ewap->u_del));
      iu2 = ((int)(u0 + this_ewap->u_del));
      iv1 = ((int)(v0 - this_ewap->v_del));
      iv2 = ((int)(v0 + this_ewap->v_del));

      if (iu1 < 0) {
        iu1 = 0;
      }
      if (iu2 >= grid_cols) {
        iu2 = (grid_cols - 1);
      }
      if (iv1 < 0) {
        iv1 = 0;
      }
      if (iv2 >= grid_rows) {
        iv2 = (grid_rows - 1);
      }

      if (iu1 < grid_cols && iu2 >= 0 && iv1 < grid_rows && iv2 >= 0) {
        got_point = 1;
        ddq = 2.0 * this_ewap->a;

        u = (iu1 - u0);
        a2up1 = (this_ewap->a * ((2.0 * u) + 1.0));
        bu = this_ewap->b * u;
        au2 = this_ewap->a * u * u;

        for (iv = iv1; iv <= iv2; iv++) {
          v = (iv - v0);
          dq = (a2up1 + (this_ewap->b * v));
          q = ((((this_ewap->c * v) + bu) * v) + au2);
          for (iu = iu1; iu <= iu2; iu++) {
            if ((q >= 0.0) && (q < this_ewap->f)) {
              iw = ((int)(q * ewaw->qfactor));
              if (iw >= ewaw->count) {
                iw = (ewaw->count - 1);
              }
              weight = (ewaw->wtab[iw]);
              grid_offset = ((iv * grid_cols) + iu);

              for (chan = 0; chan < chan_count; chan+=1) {
                this_val = ((images[chan])[swath_offset]);
                if (maximum_weight_mode) {
                  if (weight > grid_weights[chan][grid_offset] & !((this_val == img_fill) || (__isnan(this_val)))) {
                    ((grid_weights[chan])[grid_offset]) = weight;
                    ((grid_accums[chan])[grid_offset]) = (accum_type)this_val;
                  }
                } else {
                  if ((this_val != img_fill) && !(__isnan(this_val))) {
                    ((grid_weights[chan])[grid_offset]) += weight;
                    ((grid_accums[chan])[grid_offset]) += (accum_type)this_val * weight;
                  }
                }
              }
            }
            q += dq;
            dq += ddq;
          }
        }
      }
    }
  }

  /* function exit code */
  return got_point;
}


template<typename CR_TYPE, typename IMAGE_TYPE>
int compute_ewa_single(int maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows, CR_TYPE *uimg, CR_TYPE *vimg,
        IMAGE_TYPE *image, IMAGE_TYPE img_fill, accum_type *grid_accum, weight_type *grid_weight, ewa_weight *ewaw, ewa_parameters *ewap) {
  // This was originally copied from a cython C file for 32-bit float inputs (that explains some of the weird parens and other syntax
  int got_point;
  unsigned int row;
  unsigned int col;
  ewa_parameters *this_ewap;
  int iu1;
  int iu2;
  int iv1;
  int iv2;
  int iu;
  int iv;
  CR_TYPE u0;
  CR_TYPE v0;
  weight_type ddq;
  weight_type dq;
  weight_type q;
  weight_type u;
  weight_type v;
  weight_type a2up1;
  weight_type au2;
  weight_type bu;
  weight_type weight;

  int iw;
  IMAGE_TYPE this_val;
  unsigned int swath_offset;
  unsigned int grid_offset;

  got_point = 0;
  for (row = 0, swath_offset=0; row < swath_rows; row+=1) {
    for (col = 0, this_ewap = ewap; col < swath_cols; col++, this_ewap++, swath_offset++) {
      u0 = uimg[swath_offset];
      v0 = vimg[swath_offset];

      if (u0 < -this_ewap->u_del || v0 < -this_ewap->v_del || __isnan(u0) || __isnan(v0)) {
        continue;
      }

      iu1 = ((int)(u0 - this_ewap->u_del));
      iu2 = ((int)(u0 + this_ewap->u_del));
      iv1 = ((int)(v0 - this_ewap->v_del));
      iv2 = ((int)(v0 + this_ewap->v_del));

      if (iu1 < 0) {
        iu1 = 0;
      }
      if (iu2 >= grid_cols) {
        iu2 = (grid_cols - 1);
      }
      if (iv1 < 0) {
        iv1 = 0;
      }
      if (iv2 >= grid_rows) {
        iv2 = (grid_rows - 1);
      }

      if (iu1 < grid_cols && iu2 >= 0 && iv1 < grid_rows && iv2 >= 0) {
        got_point = 1;
        ddq = 2.0 * this_ewap->a;

        u = (iu1 - u0);
        a2up1 = (this_ewap->a * ((2.0 * u) + 1.0));
        bu = this_ewap->b * u;
        au2 = this_ewap->a * u * u;

        for (iv = iv1; iv <= iv2; iv++) {
          v = (iv - v0);
          dq = (a2up1 + (this_ewap->b * v));
          q = ((((this_ewap->c * v) + bu) * v) + au2);
          for (iu = iu1; iu <= iu2; iu++) {
            if ((q >= 0.0) && (q < this_ewap->f)) {
              iw = ((int)(q * ewaw->qfactor));
              if (iw >= ewaw->count) {
                iw = (ewaw->count - 1);
              }
              weight = (ewaw->wtab[iw]);
              grid_offset = ((iv * grid_cols) + iu);

              this_val = (image[swath_offset]);
              if (maximum_weight_mode) {
                if (weight > grid_weight[grid_offset] & !((this_val == img_fill) || (__isnan(this_val)))) {
                  grid_weight[grid_offset] = weight;
                  grid_accum[grid_offset] = (accum_type)this_val;
                }
              } else {
                if ((this_val != img_fill) && !(__isnan(this_val))) {
                  grid_weight[grid_offset] += weight;
                  grid_accum[grid_offset] += (accum_type)this_val * weight;
                }
              }
            }
            q += dq;
            dq += ddq;
          }
        }
      }
    }
  }

  /* function exit code */
  return got_point;
}
// Overloaded functions for specific types for `write_grid_image`
//static void write_grid_pixel(npy_uint8 *output_image, accum_type chanf) {
//  if (chanf < 0.0) {
//    *output_image = 0;
//  } else if (chanf > 255.0) {
//    *output_image = 255;
//  } else {
//    *output_image = (npy_uint8)chanf;
//  }
//}

inline void write_grid_pixel(npy_int8 *output_image, accum_type chanf) {
  if (chanf < -128.0) {
    *output_image = -128;
  } else if (chanf > 127.0) {
    *output_image = 127;
  } else {
    *output_image = (npy_int8)chanf;
  }
}

//static void write_grid_pixel(npy_uint16 *output_image, accum_type chanf) {
//  if (chanf < 0.0) {
//    *output_image = 0;
//  } else if (chanf > 65535.0) {
//    *output_image = 65535;
//  } else {
//    *output_image = (npy_uint16)chanf;
//  }
//}

//static void write_grid_pixel(npy_int16 *output_image, accum_type chanf) {
//  if (chanf < -32768.0) {
//    *output_image = -32768;
//  } else if (chanf > 32767.0) {
//    *output_image = 32767;
//  } else {
//    *output_image = (npy_int16)chanf;
//  }
//}

//static void write_grid_pixel(npy_uint32 *output_image, accum_type chanf) {
//  if (chanf < 0.0) {
//    *output_image = 0;
//  } else if (chanf > 4294967295.0) {
//    *output_image = 4294967295;
//  } else {
//    *output_image = (npy_uint32)chanf;
//  }
//}

//static void write_grid_pixel(npy_int32 *output_image, accum_type chanf) {
//  if (chanf < -2147483648.0) {
//    *output_image = -2147483648;
//  } else if (chanf > 2147483647.0) {
//    *output_image = 2147483647;
//  } else {
//    *output_image = (npy_int32)chanf;
//  }
//}

inline void write_grid_pixel(npy_float32 *output_image, accum_type chanf) {
  *output_image = (npy_float32)chanf;
}

inline void write_grid_pixel(npy_float64 *output_image, accum_type chanf) {
  *output_image = (npy_float64)chanf;
}
// End of overload functions for `write_grid_image`

template<typename GRID_TYPE>
inline accum_type get_rounding(GRID_TYPE *output_image) {
  return 0.5;
}

template<> inline accum_type get_rounding(npy_float32 *output_image) {
  return 0.0;
}

template<> inline accum_type get_rounding(npy_float64 *output_image) {
  return 0.0;
}

template<typename GRID_TYPE>
unsigned int write_grid_image(GRID_TYPE *output_image, GRID_TYPE fill, size_t grid_cols, size_t grid_rows,
        accum_type *grid_accum, weight_type *grid_weights,
        int maximum_weight_mode, weight_type weight_sum_min) {
  accum_type chanf;
  unsigned int i;
  unsigned int valid_count = 0;
  size_t grid_size = grid_cols * grid_rows;

  if (weight_sum_min <= 0.0) {
    weight_sum_min = EPSILON;
  }

  for (i=0; i < grid_size;
       i++, grid_accum++, grid_weights++, output_image++) {
    // Calculate the elliptical weighted average value for each cell (float -> not-float needs rounding)
    // The fill value for the weight and accumulation arrays is static at NaN
    if (*grid_weights < weight_sum_min || __isnan(*grid_accum)) {
      chanf = (accum_type)NPY_NANF;
    } else if (maximum_weight_mode) {
      // keep the current value
      chanf = *grid_accum;
    } else if (*grid_accum >= 0.0) {
      chanf = *grid_accum / *grid_weights + get_rounding(output_image);
    } else {
      chanf = *grid_accum / *grid_weights - get_rounding(output_image);
    }

    if (__isnan(chanf)) {
      *output_image = (GRID_TYPE)fill;
    } else {
      valid_count++;
      write_grid_pixel(output_image, chanf);
    }
  }

  return valid_count;
}



// Col/Row as 32-bit floats
template int compute_ewa_parameters<npy_float32>(size_t, size_t, npy_float32*, npy_float32*, ewa_weight*, ewa_parameters*);
template int compute_ewa<npy_float32, npy_float32>(size_t, int, size_t, size_t, size_t, size_t, npy_float32*, npy_float32*, npy_float32**, npy_float32, accum_type**, weight_type**, ewa_weight*, ewa_parameters*);
template int compute_ewa<npy_float32, npy_float64>(size_t, int, size_t, size_t, size_t, size_t, npy_float32*, npy_float32*, npy_float64**, npy_float64, accum_type**, weight_type**, ewa_weight*, ewa_parameters*);
template int compute_ewa<npy_float32, npy_int8>(size_t, int, size_t, size_t, size_t, size_t, npy_float32*, npy_float32*, npy_int8**, npy_int8, accum_type**, weight_type**, ewa_weight*, ewa_parameters*);

// Col/Row as 64-bit floats
template int compute_ewa_parameters<npy_float64>(size_t, size_t, npy_float64*, npy_float64*, ewa_weight*, ewa_parameters*);
template int compute_ewa<npy_float64, npy_float32>(size_t, int, size_t, size_t, size_t, size_t, npy_float64*, npy_float64*, npy_float32**, npy_float32, accum_type**, weight_type**, ewa_weight*, ewa_parameters*);
template int compute_ewa<npy_float64, npy_float64>(size_t, int, size_t, size_t, size_t, size_t, npy_float64*, npy_float64*, npy_float64**, npy_float64, accum_type**, weight_type**, ewa_weight*, ewa_parameters*);
template int compute_ewa<npy_float64, npy_int8>(size_t, int, size_t, size_t, size_t, size_t, npy_float64*, npy_float64*, npy_int8**, npy_int8, accum_type**, weight_type**, ewa_weight*, ewa_parameters*);

// Single channel
// Col/Row as 32-bit floats
template int compute_ewa_single<npy_float32, npy_float32>(int, size_t, size_t, size_t, size_t, npy_float32*, npy_float32*, npy_float32*, npy_float32, accum_type*, weight_type*, ewa_weight*, ewa_parameters*);
template int compute_ewa_single<npy_float32, npy_float64>(int, size_t, size_t, size_t, size_t, npy_float32*, npy_float32*, npy_float64*, npy_float64, accum_type*, weight_type*, ewa_weight*, ewa_parameters*);
template int compute_ewa_single<npy_float32, npy_int8>(int, size_t, size_t, size_t, size_t, npy_float32*, npy_float32*, npy_int8*, npy_int8, accum_type*, weight_type*, ewa_weight*, ewa_parameters*);

// Col/Row as 64-bit floats
template int compute_ewa_single<npy_float64, npy_float32>(int, size_t, size_t, size_t, size_t, npy_float64*, npy_float64*, npy_float32*, npy_float32, accum_type*, weight_type*, ewa_weight*, ewa_parameters*);
template int compute_ewa_single<npy_float64, npy_float64>(int, size_t, size_t, size_t, size_t, npy_float64*, npy_float64*, npy_float64*, npy_float64, accum_type*, weight_type*, ewa_weight*, ewa_parameters*);
template int compute_ewa_single<npy_float64, npy_int8>(int, size_t, size_t, size_t, size_t, npy_float64*, npy_float64*, npy_int8*, npy_int8, accum_type*, weight_type*, ewa_weight*, ewa_parameters*);


// Output Grid types
template unsigned int write_grid_image<npy_float32>(npy_float32*, npy_float32, size_t, size_t, accum_type*, weight_type*, int, weight_type);
template unsigned int write_grid_image<npy_float64>(npy_float64*, npy_float64, size_t, size_t, accum_type*, weight_type*, int, weight_type);
template unsigned int write_grid_image<npy_int8>(npy_int8*, npy_int8, size_t, size_t, accum_type*, weight_type*, int, weight_type);
