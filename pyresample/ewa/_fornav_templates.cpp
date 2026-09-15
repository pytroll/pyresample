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

  // Always leave the struct in a state that deinitialize_weight() can handle
  ewaw->wtab = NULL;

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

  // The table has one entry more than weight_count so compute_ewa*() can index it with
  // `wtab[(int)(q * qfactor)]` and no bounds check. With q < qmax and qfactor = count / qmax
  // the index is at most `count`, and only reaches it through float rounding; that last entry
  // is a copy of the entry before it.
  ewaw->wtab = (weight_type *)calloc(weight_count + 1, sizeof(weight_type));
  if (!ewaw->wtab) {
    return -1;
  }

  ewaw->count = weight_count;
  ewaw->min = weight_min;
  ewaw->distance_max = weight_distance_max;
  ewaw->delta_max = weight_delta_max;
  ewaw->sum_min = weight_sum_min;

  ewaw->qmax = ewaw->distance_max * ewaw->distance_max;
  ewaw->alpha = -log(ewaw->min) / ewaw->qmax;
  wptr = ewaw->wtab;
  for (idx=0; idx < weight_count; idx++) {
    wptr[idx] = exp(-ewaw->alpha * ewaw->qmax * idx / (ewaw->count - 1));
  }
  wptr[weight_count] = wptr[weight_count - 1];

  ewaw->qfactor = ewaw->count / ewaw->qmax;
  return 0;
}

void deinitialize_weight(ewa_weight *ewaw) {
  if (ewaw->wtab) {
    free(ewaw->wtab);
    ewaw->wtab = NULL;
  }
}

void deinitialize_grids(size_t chan_count, void **grids) {
  unsigned int i;
  if (!grids) {
    return;
  }
  for (i = 0; i < chan_count; i++) {
    if (grids[i]) {
      free(grids[i]);
    }
  }
  free(grids);
}

// Allocate chan_count zeroed grids of elem_size * grid_cols * grid_rows bytes.
// On any failure everything allocated so far is released and NULL is returned.
static void **initialize_grids(size_t chan_count, size_t grid_cols, size_t grid_rows, size_t elem_size) {
  void **grids = (void **)calloc(chan_count, sizeof(void *));
  unsigned int i;

  if (!grids) {
    return NULL;
  }
  for (i=0; i < chan_count; i++) {
    grids[i] = calloc(grid_cols * grid_rows, elem_size);
    if (!grids[i]) {
      deinitialize_grids(chan_count, grids);
      return NULL;
    }
  }

  return grids;
}

accum_type **initialize_grid_accums(size_t chan_count, size_t grid_cols, size_t grid_rows) {
  return (accum_type **)initialize_grids(chan_count, grid_cols, grid_rows, sizeof(accum_type));
}

weight_type **initialize_grid_weights(size_t chan_count, size_t grid_cols, size_t grid_rows) {
  return (weight_type **)initialize_grids(chan_count, grid_cols, grid_rows, sizeof(weight_type));
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
    if (__isnan(ux) || __isnan(vx) || __isnan(uy) || __isnan(vy)) {
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

// Bounding box of grid cells (inclusive) that a swath pixel at (u0, v0) can touch:
// [u0 - u_del, u0 + u_del] x [v0 - v_del, v0 + v_del], truncated to ints and clamped to the
// grid. Returns 0 without touching the outputs if that box cannot overlap the grid at all.
//
// The overlap test is done on the floating point bounds rather than on the clamped ints
// because most swath pixels fail it: in the dask path every source block is offered to every
// target chunk it might overlap, so a large share of coordinates land far outside this
// chunk's grid. Those pixels then cost four comparisons and nothing else. Writing the test
// as a positive condition also makes NaN coordinates fail it without an explicit isnan(),
// and keeps huge values away from the float->int cast, whose result would be undefined.
template<typename CR_TYPE>
static inline int compute_bbox(CR_TYPE u0, CR_TYPE v0, ewa_param_type u_del, ewa_param_type v_del,
        CR_TYPE grid_cols_f, CR_TYPE grid_rows_f, size_t grid_cols, size_t grid_rows,
        int *iu1, int *iu2, int *iv1, int *iv2) {
  const CR_TYPE u_lo = u0 - u_del;
  const CR_TYPE v_lo = v0 - v_del;
  if (!(u0 >= -u_del && v0 >= -v_del && u_lo < grid_cols_f && v_lo < grid_rows_f)) {
    return 0;
  }
  // Because u0 >= -u_del, u0 + u_del >= 0, and because u_lo < grid_cols, (int)u_lo < grid_cols
  // (same for v), so after clamping the box is a non-empty range of grid cells.
  *iu1 = (int)u_lo;
  *iu2 = (int)(u0 + u_del);
  *iv1 = (int)v_lo;
  *iv2 = (int)(v0 + v_del);
  if (*iu1 < 0) {
    *iu1 = 0;
  }
  if (*iu2 >= (int)grid_cols) {
    *iu2 = (int)grid_cols - 1;
  }
  if (*iv1 < 0) {
    *iv1 = 0;
  }
  if (*iv2 >= (int)grid_rows) {
    *iv2 = (int)grid_rows - 1;
  }
  return 1;
}

template<typename IMAGE_TYPE>
static inline int is_valid_pixel(IMAGE_TYPE this_val, IMAGE_TYPE img_fill) {
  return !(this_val == img_fill || __isnan(this_val));
}

// compute_ewa*() below: for every swath pixel, walk the grid cells in its bounding box and
// add the pixel's value, scaled by the elliptical weight `wtab[q * qfactor]` of the cell,
// into grid_accum, and the weight itself into grid_weight (or, in maximum weight mode, keep
// the value with the single largest weight). write_grid_image() turns the two grids into
// the output image afterwards.
//
// Notes on how the loops are written, since this is the hot path of EWA resampling:
//
// - The ellipse parameters `a`, `b`, `c`, `f` and the weight table fields are float, and so
//   are the grid arrays the loops store into. Under C++ aliasing rules a store through a
//   `float *` may modify any other float, so if the loops read those parameters through the
//   struct pointers the compiler has to reload them from memory after every grid store.
//   They are therefore copied into local variables before the loops, and the pointer
//   arguments are declared FORNAV_RESTRICT, which promises that the grid arrays, the image
//   and the coordinate arrays do not overlap.
// - Whether a pixel's value is valid (not the fill value, not NaN) does not depend on the
//   grid cell, so it is checked once per swath pixel, before the bounding box loops.
// - maximum_weight_mode is a template parameter so each mode gets its own specialised inner
//   loop instead of a runtime branch per grid cell.

template<bool MAX_WEIGHT_MODE, typename CR_TYPE, typename IMAGE_TYPE>
static int compute_ewa_impl(size_t chan_count,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        const CR_TYPE *FORNAV_RESTRICT uimg, const CR_TYPE *FORNAV_RESTRICT vimg,
        IMAGE_TYPE **images, IMAGE_TYPE img_fill, accum_type **grid_accums, weight_type **grid_weights,
        const ewa_weight *ewaw, const ewa_parameters *ewap) {
  const weight_type qfactor = ewaw->qfactor;
  const weight_type *FORNAV_RESTRICT wtab = ewaw->wtab;
  const CR_TYPE grid_cols_f = (CR_TYPE)grid_cols;
  const CR_TYPE grid_rows_f = (CR_TYPE)grid_rows;
  int got_point = 0;
  unsigned int row;
  unsigned int col;
  unsigned int swath_offset;
  size_t chan;
  size_t n_valid;
  int iu1;
  int iu2;
  int iv1;
  int iv2;
  int iu;
  int iv;

  // Filled per swath pixel with the channels whose value is valid (not fill, not NaN) and
  // those values, so the inner loop only visits channels that contribute.
  size_t *valid_chans = (size_t *)malloc(chan_count * sizeof(size_t));
  accum_type *valid_vals = (accum_type *)malloc(chan_count * sizeof(accum_type));
  if (!valid_chans || !valid_vals) {
    free(valid_chans);
    free(valid_vals);
    return -1;
  }

  for (row = 0, swath_offset = 0; row < swath_rows; row++) {
    const ewa_parameters *this_ewap = ewap;
    for (col = 0; col < swath_cols; col++, this_ewap++, swath_offset++) {
      const CR_TYPE u0 = uimg[swath_offset];
      const CR_TYPE v0 = vimg[swath_offset];

      if (!compute_bbox(u0, v0, this_ewap->u_del, this_ewap->v_del, grid_cols_f, grid_rows_f,
                        grid_cols, grid_rows, &iu1, &iu2, &iv1, &iv2)) {
        continue;
      }
      got_point = 1;

      n_valid = 0;
      for (chan = 0; chan < chan_count; chan++) {
        const IMAGE_TYPE this_val = images[chan][swath_offset];
        if (is_valid_pixel(this_val, img_fill)) {
          valid_chans[n_valid] = chan;
          valid_vals[n_valid] = (accum_type)this_val;
          n_valid++;
        }
      }
      if (n_valid == 0) {
        continue;
      }

      const ewa_param_type a = this_ewap->a;
      const ewa_param_type b = this_ewap->b;
      const ewa_param_type c = this_ewap->c;
      const ewa_param_type f = this_ewap->f;
      const weight_type ddq = 2.0 * a;
      const weight_type u = (iu1 - u0);
      const weight_type a2up1 = (a * ((2.0 * u) + 1.0));
      const weight_type bu = b * u;
      const weight_type au2 = a * u * u;

      for (iv = iv1; iv <= iv2; iv++) {
        const weight_type v = (iv - v0);
        const size_t row_offset = (size_t)iv * grid_cols;
        weight_type dq = (a2up1 + (b * v));
        weight_type q = ((((c * v) + bu) * v) + au2);
        for (iu = iu1; iu <= iu2; iu++) {
          if ((q >= 0.0) && (q < f)) {
            const weight_type weight = wtab[(int)(q * qfactor)];
            const size_t grid_offset = row_offset + iu;

            for (chan = 0; chan < n_valid; chan++) {
              weight_type *FORNAV_RESTRICT grid_weight = grid_weights[valid_chans[chan]];
              accum_type *FORNAV_RESTRICT grid_accum = grid_accums[valid_chans[chan]];
              const accum_type val = valid_vals[chan];
              if (MAX_WEIGHT_MODE) {
                if (weight > grid_weight[grid_offset]) {
                  grid_weight[grid_offset] = weight;
                  grid_accum[grid_offset] = val;
                }
              } else {
                grid_weight[grid_offset] += weight;
                grid_accum[grid_offset] += val * weight;
              }
            }
          }
          q += dq;
          dq += ddq;
        }
      }
    }
  }

  free(valid_chans);
  free(valid_vals);
  return got_point;
}

template<typename CR_TYPE, typename IMAGE_TYPE>
int compute_ewa(size_t chan_count, int maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows, CR_TYPE *uimg, CR_TYPE *vimg,
        IMAGE_TYPE **images, IMAGE_TYPE img_fill, accum_type **grid_accums, weight_type **grid_weights, ewa_weight *ewaw, ewa_parameters *ewap) {
  if (maximum_weight_mode) {
    return compute_ewa_impl<true, CR_TYPE, IMAGE_TYPE>(chan_count, swath_cols, swath_rows, grid_cols, grid_rows,
        uimg, vimg, images, img_fill, grid_accums, grid_weights, ewaw, ewap);
  }
  return compute_ewa_impl<false, CR_TYPE, IMAGE_TYPE>(chan_count, swath_cols, swath_rows, grid_cols, grid_rows,
      uimg, vimg, images, img_fill, grid_accums, grid_weights, ewaw, ewap);
}


template<bool MAX_WEIGHT_MODE, typename CR_TYPE, typename IMAGE_TYPE>
static int compute_ewa_single_impl(
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows,
        const CR_TYPE *FORNAV_RESTRICT uimg, const CR_TYPE *FORNAV_RESTRICT vimg,
        const IMAGE_TYPE *FORNAV_RESTRICT image, IMAGE_TYPE img_fill,
        accum_type *FORNAV_RESTRICT grid_accum, weight_type *FORNAV_RESTRICT grid_weight,
        const ewa_weight *ewaw, const ewa_parameters *ewap) {
  const weight_type qfactor = ewaw->qfactor;
  const weight_type *FORNAV_RESTRICT wtab = ewaw->wtab;
  const CR_TYPE grid_cols_f = (CR_TYPE)grid_cols;
  const CR_TYPE grid_rows_f = (CR_TYPE)grid_rows;
  int got_point = 0;
  unsigned int row;
  unsigned int col;
  unsigned int swath_offset;
  int iu1;
  int iu2;
  int iv1;
  int iv2;
  int iu;
  int iv;

  for (row = 0, swath_offset = 0; row < swath_rows; row++) {
    const ewa_parameters *this_ewap = ewap;
    for (col = 0; col < swath_cols; col++, this_ewap++, swath_offset++) {
      const CR_TYPE u0 = uimg[swath_offset];
      const CR_TYPE v0 = vimg[swath_offset];

      if (!compute_bbox(u0, v0, this_ewap->u_del, this_ewap->v_del, grid_cols_f, grid_rows_f,
                        grid_cols, grid_rows, &iu1, &iu2, &iv1, &iv2)) {
        continue;
      }
      got_point = 1;

      const IMAGE_TYPE this_val = image[swath_offset];
      if (!is_valid_pixel(this_val, img_fill)) {
        continue;
      }

      const accum_type val = (accum_type)this_val;
      const ewa_param_type a = this_ewap->a;
      const ewa_param_type b = this_ewap->b;
      const ewa_param_type c = this_ewap->c;
      const ewa_param_type f = this_ewap->f;
      const weight_type ddq = 2.0 * a;
      const weight_type u = (iu1 - u0);
      const weight_type a2up1 = (a * ((2.0 * u) + 1.0));
      const weight_type bu = b * u;
      const weight_type au2 = a * u * u;

      for (iv = iv1; iv <= iv2; iv++) {
        const weight_type v = (iv - v0);
        const size_t row_offset = (size_t)iv * grid_cols;
        weight_type dq = (a2up1 + (b * v));
        weight_type q = ((((c * v) + bu) * v) + au2);
        for (iu = iu1; iu <= iu2; iu++) {
          if ((q >= 0.0) && (q < f)) {
            const weight_type weight = wtab[(int)(q * qfactor)];
            const size_t grid_offset = row_offset + iu;

            if (MAX_WEIGHT_MODE) {
              if (weight > grid_weight[grid_offset]) {
                grid_weight[grid_offset] = weight;
                grid_accum[grid_offset] = val;
              }
            } else {
              grid_weight[grid_offset] += weight;
              grid_accum[grid_offset] += val * weight;
            }
          }
          q += dq;
          dq += ddq;
        }
      }
    }
  }

  return got_point;
}

template<typename CR_TYPE, typename IMAGE_TYPE>
int compute_ewa_single(int maximum_weight_mode,
        size_t swath_cols, size_t swath_rows, size_t grid_cols, size_t grid_rows, CR_TYPE *uimg, CR_TYPE *vimg,
        IMAGE_TYPE *image, IMAGE_TYPE img_fill, accum_type *grid_accum, weight_type *grid_weight, ewa_weight *ewaw, ewa_parameters *ewap) {
  if (maximum_weight_mode) {
    return compute_ewa_single_impl<true, CR_TYPE, IMAGE_TYPE>(swath_cols, swath_rows, grid_cols, grid_rows,
        uimg, vimg, image, img_fill, grid_accum, grid_weight, ewaw, ewap);
  }
  return compute_ewa_single_impl<false, CR_TYPE, IMAGE_TYPE>(swath_cols, swath_rows, grid_cols, grid_rows,
      uimg, vimg, image, img_fill, grid_accum, grid_weight, ewaw, ewap);
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
