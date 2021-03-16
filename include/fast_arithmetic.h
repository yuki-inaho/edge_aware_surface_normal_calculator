#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// from https://gist.github.com/volkansalma/2972237
//  or  http://lists.apple.com/archives/perfoptimization-dev/2005/Jan/msg00051.html
// |error| < 0.005
#define PI_FLOAT     3.14159265f
#define PIBY2_FLOAT  1.5707963f


float fast_atan2f_1(float y, float x)
{
    if (x == 0.0f)
    {
        if (y > 0.0f)
            return PIBY2_FLOAT;
        if (y == 0.0f)
            return 0.0f;
        return -PIBY2_FLOAT;
    }
    float atan;
    float z = y / x;
    if (fabsf(z) < 1.0f)
    {
        atan = z / (1.0f + 0.28f * z * z);
        if (x < 0.0f)
        {
            if (y < 0.0f)
                return atan - PI_FLOAT;
            return atan + PI_FLOAT;
        }
    }
    else
    {
        atan = PIBY2_FLOAT - z / (z * z + 0.28f);
        if (y < 0.0f)
            return atan - PI_FLOAT;
    }
    return atan;
}

float fast_atan2f_2(float y, float x)
{
    //http://pubs.opengroup.org/onlinepubs/009695399/functions/atan2.html
    //Volkan SALMA

    const float ONEQTR_PI = M_PI / 4.0;
    const float THRQTR_PI = 3.0 * M_PI / 4.0;
    float r, angle;
    float abs_y = fabs(y) + 1e-10f; // kludge to prevent 0/0 condition
    if (x < 0.0f)
    {
        r = (x + abs_y) / (abs_y - x);
        angle = THRQTR_PI;
    }
    else
    {
        r = (x - abs_y) / (x + abs_y);
        angle = ONEQTR_PI;
    }
    angle += (0.1963f * r * r - 0.9817f) * r;
    if (y < 0.0f)
        return (-angle); // negate if in quad III or IV
    else
        return (angle);
}

float fast_arccosf(float x)
{
    float x2 = x * x;
    float x4 = x2 * x2;
    return (M_PI / 2.0 - (x + 1. / 6. * x * x2 + 3. / 40. * x * x4));
}

/* Assumes that float is in the IEEE 754 single precision floating point format
	 * and that int is 32 bits.
	 * http://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Approximations_that_depend_on_the_floating_point_representation */
float fast_sqrt(float z)
{
    int val_int = *(int *)&z; /* Same bits, but as an int */
    /*
	     * To justify the following code, prove that
	     *
	     * ((((val_int / 2^m) - b) / 2) + b) * 2^m = ((val_int - 2^m) / 2) + ((b + 1) / 2) * 2^m)
	     *
	     * where
	     *
	     * b = exponent bias
	     * m = number of mantissa bits
	     *
	     * .
	     */

    val_int -= 1 << 23; /* Subtract 2^m. */
    val_int >>= 1;      /* Divide by 2. */
    val_int += 1 << 29; /* Add ((b + 1) / 2) * 2^m. */

    return *(float *)&val_int; /* Interpret again as float */
}