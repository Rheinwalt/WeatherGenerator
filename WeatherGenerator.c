#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>

#define VERSION "0.3"

typedef struct Day Day;

struct Day {
	Day *next;
	double *wet, *dry;
	unsigned int wetsize, drysize;
};

int
compare(const void *a, const void *b) {
    double x, y;

    x = *(double *)a;
    y = *(double *)b;
    if(x > y)
        return 1;
    else if(x < y)
        return -1;
    return 0;
}

int
initrng(void) {
	int i, rdev;
	long int seed;

	rdev = open("/dev/urandom", O_RDONLY);
	if(!rdev) {
		PyErr_SetString(PyExc_RuntimeError, "error: cannot open /dev/urandom\n");
		return 0;
	}
	read(rdev, &seed, sizeof(long int));
	close(rdev);
	srand48(seed);
	for(i = 0; i < 10; i++)
		drand48();
	return 1;
}

static PyObject *
WeatherGenerator_GenerateEvents(PyObject *self, PyObject* args) {
    PyObject *dryarg, *wetarg;
    PyArrayObject *dryobj, *wetobj, *events;
	double *dry, *wet;
	unsigned int i, ii, k, ki, l, m, *e;
	unsigned int tlen, drylen, wetlen;
	npy_intp *dim;

    if(!PyArg_ParseTuple(args, "OOI", &dryarg, &wetarg, &tlen))
		return NULL;
    dryobj = (PyArrayObject *) PyArray_ContiguousFromObject(dryarg,
							   PyArray_DOUBLE, 1, 1);
    wetobj = (PyArrayObject *) PyArray_ContiguousFromObject(wetarg,
							   PyArray_DOUBLE, 1, 1);
    if(!dryobj || !wetobj)
        return NULL;
	dry = (double *) dryobj->data;
	wet = (double *) wetobj->data;
	drylen = dryobj->dimensions[0];
	wetlen = wetobj->dimensions[0];
	dim = malloc(sizeof(npy_intp));
	dim[0] = tlen;
	events = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
	free(dim);
	if(!events) {
		PyErr_SetString(PyExc_MemoryError, "Cannot create (PyArrayObject *)events for output of GenerateEvents().");
		return NULL;
	}
	e = (unsigned int *) events->data;
	if(!initrng())
		return NULL;
	e[1] = 1;
	l = 2;
	ki = 0;
	ii = 1;
	while(l < tlen) {
		k = ki;
		i = ii;
		if(drand48() < dry[k++]) {
			while(k < drylen)
				if(dry[k++] <= drand48())
					break;
			l += k - ki - 1;
			if(tlen <= l)
				break;
			ii = 1;
			ki = 0;
			e[l++] = 1;
		} else {
			i++;
			while(i < wetlen)
				if(wet[i++] <= drand48())
					break;
			for(m = ii; m < i - 1; m++) {
				e[l++] = 1;
				if(tlen <= l)
					break;
			}
			ki = 1;
			ii = 1;
			l++;
		}
	}
	Py_DECREF(dryobj);
	Py_DECREF(wetobj);
	return PyArray_Return(events);
}

static PyObject *
WeatherGenerator_EventThresholds(PyObject *self, PyObject* args) {
    PyObject *arg;
    PyArrayObject *precip, *threshs;
	double *p, *t, *b, ti, perc;
	unsigned int tlen, period, bin;
	unsigned int i, k, l, n, m;
	npy_intp *dim;

	period = 365;
	bin = 3;
    if(!PyArg_ParseTuple(args, "Od|II", &arg, &perc, &bin, &period))
		return NULL;
	if(perc <= 0 || 1 <= perc) {
		PyErr_SetString(PyExc_RuntimeError, "Percentile has to be given on the range (0, 1).");
		return NULL;
	}
    precip = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
							   PyArray_DOUBLE, 1, 1);
    if(!precip)
        return NULL;
	p = (double *) precip->data;
	tlen = precip->dimensions[0];
	dim = malloc(sizeof(npy_intp));
	dim[0] = tlen;
	threshs = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
	free(dim);
	if(!threshs) {
		PyErr_SetString(PyExc_MemoryError, "Cannot create (PyArrayObject *)threshs for output of EventThresholds().");
		return NULL;
	}
	t = (double *) threshs->data;
	n = (2 * bin + 1) * tlen / period + 10;
	b = malloc(n * sizeof(double));
	if(!b) {
		PyErr_SetString(PyExc_MemoryError, "Cannot estimate percentile thresholds in EventThresholds().");
		return NULL;
	}
	for(i = 0; i < bin; i++) {
		m = 0;
		for(k = 0; k < i + bin + 1; k++) {
			for(l = k; l < tlen; l += period)
				b[m++] = p[l];
		}
		for(k = i + period - bin; k < period; k++) {
			for(l = k; l < tlen; l += period)
				b[m++] = p[l];
		}
		qsort(b, m, sizeof(double), compare);
		t[i] = b[(int)(m * perc)];
	}
	for(i = bin; i < period; i++) {
		m = 0;
		for(k = i - bin; k < i + bin + 1; k++) {
			for(l = k; l < tlen; l += period)
				b[m++] = p[l];
		}
		qsort(b, m, sizeof(double), compare);
		t[i] = b[(int)(m * perc)];
	}
	free(b);
	for(i = 0; i < period; i++) {
		ti = t[i];
		for(l = i + period; l < tlen; l += period)
			t[l] = ti;
	}
	Py_DECREF(precip);
	return PyArray_Return(threshs);
}

static PyObject *
WeatherGenerator_Synthesize(PyObject *self, PyObject* args) {
	PyObject *prarg, *evarg, *tharg;
	PyArrayObject *precip, *events, *threshs, *synth;
	double *p, *s, *t;
	unsigned int tlen, elen, period, bin, *e;
	unsigned int i, k, l, n, mw, md;
	int skip;
	npy_intp *dim;
	Day *start, *new, *day;

	period = 365;
	bin = 3;
	skip = 0;
    if(!PyArg_ParseTuple(args, "OOO|i", &prarg, &evarg, &tharg, &skip))
		return NULL;
    precip = (PyArrayObject *) PyArray_ContiguousFromObject(prarg,
							   PyArray_DOUBLE, 1, 1);
    events = (PyArrayObject *) PyArray_ContiguousFromObject(evarg,
							   PyArray_UINT, 1, 1);
    threshs = (PyArrayObject *) PyArray_ContiguousFromObject(tharg,
							   PyArray_DOUBLE, 1, 1);
    if(!precip || !events || !threshs)
        return NULL;
	p = (double *) precip->data;
	tlen = precip->dimensions[0];
	e = (unsigned int *) events->data;
	elen = events->dimensions[0];
	t = (double *) threshs->data;
	if(threshs->dimensions[0] < period) {
		PyErr_SetString(PyExc_RuntimeError, "Sequence of event thresholds to short.");
		return NULL;
	}
	/* we realloc anyways, so no special case treats for n */
	n = (2 * bin + 1) * tlen / period + 10;
	for(i = 0; i < bin; i++) {
		new = malloc(sizeof(Day));
		new->wet = malloc(n * sizeof(double));
		new->dry = malloc(n * sizeof(double));
		day->next = new;
		day = new;
		if(!i) {
			start = malloc(sizeof(Day));
			start->next = day;
		}
		mw = md = 0;
		for(k = 0; k < i + bin + 1; k++) {
			for(l = k; l < tlen; l += period) {
				if(p[l] > t[i])
					day->wet[mw++] = p[l];
				else
					day->dry[md++] = p[l];
			}
		}
		for(k = i + period - bin; k < period; k++) {
			for(l = k; l < tlen; l += period) {
				if(p[l] > t[i])
					day->wet[mw++] = p[l];
				else
					day->dry[md++] = p[l];
			}
		}
		day->wetsize = mw;
		day->drysize = md;
		day->wet = realloc(day->wet, mw * sizeof(double));
		day->dry = realloc(day->dry, md * sizeof(double));
	}
	for(i = bin; i < period; i++) {
		new = malloc(sizeof(Day));
		new->wet = malloc(n * sizeof(double));
		new->dry = malloc(n * sizeof(double));
		day->next = new;
		day = new;
		mw = md = 0;
		for(k = i - bin; k < i + bin + 1; k++) {
			for(l = k; l < tlen; l += period) {
				if(p[l] > t[i])
					day->wet[mw++] = p[l];
				else
					day->dry[md++] = p[l];
			}
		}
		day->wetsize = mw;
		day->drysize = md;
		day->wet = realloc(day->wet, mw * sizeof(double));
		day->dry = realloc(day->dry, md * sizeof(double));
		/* full dry/wet period fix */
		if(mw == 0) {
			day->wetsize = md;
			day->wet = day->dry;
		} else if(md == 0) {
			day->drysize = mw;
			day->dry = day->wet;
		}
	}
	day->next = start->next;
	Py_DECREF(precip);
	Py_DECREF(threshs);

	/* ready for output */
	dim = malloc(sizeof(npy_intp));
	dim[0] = elen;
	synth = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
	free(dim);
	if(!synth) {
		PyErr_SetString(PyExc_MemoryError, "Cannot create (PyArrayObject *)synth for output of Synthesize().");
		return NULL;
	}
	s = (double *) synth->data;
	day = start->next;
	for(l = 0; l < skip; l++)
		day = day->next;
	for(l = 0; l < elen; l++) {
		if(e[l])
			s[l] = day->wet[(unsigned int)(day->wetsize * drand48())];
		else
			s[l] = day->dry[(unsigned int)(day->drysize * drand48())];
		day = day->next;
	}
	Py_DECREF(events);
	/* yes, we should free the circle of days here .. */
	return PyArray_Return(synth);
}

static PyObject *
WeatherGenerator_EstimateProbabilities(PyObject *self, PyObject* args) {
    PyObject *arg;
    PyArrayObject *events, *probs;
	double *p;
	unsigned int *e, *bursts, *pmf;
	unsigned int kmax, len, olen, gap;
	unsigned int nbursts, plength;
	unsigned int n, i, k, l;
	npy_intp *dim;

    if(!PyArg_ParseTuple(args, "O", &arg))
		return NULL;
    events = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
							   PyArray_UINT, 1, 1);
    if(!events)
        return NULL;
	e = (unsigned int *) events->data;
	n = events->dimensions[0];
	/* count bursts */
	len = 1024;
	bursts = malloc(len * sizeof(unsigned int));
	if(!bursts) {
		PyErr_SetString(PyExc_MemoryError, "Cannot alloc memory for bursts in EstimateProbabilities().\n");
		return NULL;
	}
	k = gap = 0;
	for(i = 0; i < n; i++) {
		if(e[i]) {
			gap++;
		} else {
			bursts[k++] = gap;
			gap = 0;
			if(k == len) {
				len += 1024;
				bursts = realloc(bursts, len * sizeof(unsigned int));
				if(!bursts) {
					PyErr_SetString(PyExc_MemoryError, "Cannot alloc memory for bursts in EstimateProbabilities().\n");
					return NULL;
				}
			}
		}
	}
	nbursts = k;
	bursts = realloc(bursts, nbursts * sizeof(unsigned int));
	Py_DECREF(events);
	/* estimate pmf from bursts */
	len = 64;
	pmf = calloc(len, sizeof(unsigned int));
	if(!pmf) {
		PyErr_SetString(PyExc_MemoryError, "Cannot alloc memory for the pmf in EstimateProbabilities().\n");
		return NULL;
	}
	kmax = 0;
	for(i = 0; i < nbursts; i++) {
		k = bursts[i];
		if(k > kmax)
			kmax = k;
		if(k >= len) {
			olen = len;
			len = k + 64;
			pmf = realloc(pmf, len * sizeof(unsigned int));
			if(!pmf) {
				PyErr_SetString(PyExc_MemoryError, "Cannot alloc memory for the pmf in EstimateProbabilities().\n");
				return NULL;
			}
			for(l = olen; l < len; l++)
				pmf[l] = 0;
		}
		pmf[k]++;
	}
	plength = kmax + 1;
	pmf = realloc(pmf, plength * sizeof(unsigned int));
	free(bursts);
	/* compute incremental probs from pmf */
	dim = malloc(sizeof(npy_intp));
	dim[0] = plength;
	probs = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
	free(dim);
	if(!probs) {
		PyErr_SetString(PyExc_MemoryError, "Cannot create (PyArrayObject *)probs for output of EstimateProbabilities().");
		return NULL;
	}
	p = (double *)probs->data;
	i = plength - 1;
	p[i] = 0;
	k = pmf[i];
	for(i = plength - 1; i > 0; i--) {
		p[i - 1] = (double)k / (k + pmf[i - 1]);
		k += pmf[i - 1];
	}
	free(pmf);
	return PyArray_Return(probs);
}

static PyObject *
WeatherGenerator_Bursts(PyObject *self, PyObject* args) {
    PyObject *arg;
    PyArrayObject *events, *bursts;
	unsigned int *e, *b, *u;
	unsigned int len, gap;
	unsigned int nbursts;
	unsigned int n, i, k, l;
	npy_intp *dim;

    if(!PyArg_ParseTuple(args, "O", &arg))
		return NULL;
    events = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
							   PyArray_UINT, 1, 1);
    if(!events)
        return NULL;
	e = (unsigned int *) events->data;
	n = events->dimensions[0];
	/* count bursts */
	len = 1024;
	b = malloc(len * sizeof(unsigned int));
	if(!b) {
		PyErr_SetString(PyExc_MemoryError, "Cannot alloc memory for bursts in EstimateProbabilities().\n");
		return NULL;
	}
	k = gap = 0;
	for(i = 0; i < n; i++) {
		if(e[i]) {
			gap++;
		} else if(gap) {
			b[k++] = gap;
			gap = 0;
			if(k == len) {
				len += 1024;
				b = realloc(b, len * sizeof(unsigned int));
				if(!b) {
					PyErr_SetString(PyExc_MemoryError, "Cannot alloc memory for bursts in EstimateProbabilities().\n");
					return NULL;
				}
			}
		}
	}
	nbursts = k;
	b = realloc(b, nbursts * sizeof(unsigned int));
	Py_DECREF(events);
	dim = malloc(sizeof(npy_intp));
	dim[0] = nbursts;
	bursts = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
	free(dim);
	if(!bursts) {
		PyErr_SetString(PyExc_MemoryError, "Cannot create (PyArrayObject *)probs for output of EstimateProbabilities().");
		return NULL;
	}
	u = (unsigned int *)bursts->data;
	for(k = 0; k < nbursts; k++)
		u[k] = b[k];
	free(b);
	return PyArray_Return(bursts);
}

static PyMethodDef WeatherGenerator_methods[] = {
	{"GenerateEvents", WeatherGenerator_GenerateEvents, METH_VARARGS,
	 "events = GenerateEvents(tlen, dryprobs, wetprobs)\n\nGenerates an event series of length tlen for incremental probability sequences dryprobs and wetprobs.\n\nParameters\n----------\ntlen : unsigned integer\nLength of generated event series.\n\ndryprobs : array_like, double\nSequence of incremental probabilities for dry periods (no events).\n\nwetprobs : array_like, double\nSequence of incremental probabilities for wet periods (events).\n\nReturns\n-------\nevents : ndarray, uint32\nA new flat unsigned integer event series array with ones for events.\n"},
	{"Bursts", WeatherGenerator_Bursts, METH_VARARGS,
	 "bursts = Bursts(events)\n\nCounts event bursts in the given event series.\n\nParameters\n----------\nevents : array_like, uint32\nSequence of ones and zeros with ones for events.\n\nReturns\n-------\nbursts : ndarray, uint32\nThe sequence of bursts in the event series.\n"},
	{"EstimateProbabilities", WeatherGenerator_EstimateProbabilities, METH_VARARGS,
	 "probs = EstimateProbabilities(events)\n\nEstimates incremental probabilities from event series. For ones as wet (dry) days this estimates the so called wetprobs (dryprobs) as parameters for GenerateEvents().\n\nParameters\n----------\nevents : array_like, uint32\nSequence of ones and zeros with ones for events.\n\nReturns\n-------\nprobs : ndarray, double\nThe sequence of incremental probabilities.\n"},
	{"EventThresholds", WeatherGenerator_EventThresholds, METH_VARARGS,
	 "threshs = EventThresholds(precip, perc)\n\nEstimates seasonal percentile thresholds. The thresholds for each day of the year are estimated from a weekly window centered around that day.\n\nParameters\n----------\nprecip : array_like, double\nPrecipitation time series.\n\nperc : double\nPercentile as a number between zero and one.\n\nReturns\n-------\nthreshs : ndarray, double\nThe sequence of thresholds.\n"},
	{"Synthesize", WeatherGenerator_Synthesize, METH_VARARGS,
	 "synth = Synthesize(precip, events, threshs)\n\nSynthesizes a precipitation time series for an event series that was generated by GenerateEvents(). The resulting precipitation time series has the same length as the input event series.\n\nParameters\n----------\nprecip : array_like, double\nThe precipitation time series that was used for the estimation of incremental probabilities.\n\nevents : array_like, uint32\nThe input event series as generated by GenerateEvents().\n\nthreshs : array_like, double\nThe sequence of thresholds by which >>events<< were compiled from >>precip<<.\n\nReturns\n-------\nthreshs : ndarray, double\nThe precipitation time series.\n"},
    {NULL, NULL, 0, NULL}
};

void
initWeatherGenerator(void) {
	PyObject *m;
	PyObject *v;

	v = Py_BuildValue("s", VERSION);
    PyImport_AddModule("WeatherGenerator");
    m = Py_InitModule3("WeatherGenerator", WeatherGenerator_methods,
    "Yet another stochastic precipitation generator.");
    PyModule_AddObject(m, "__version__", v);
    import_array();
}

int
main(int argc, char **argv) {
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initWeatherGenerator();
    Py_Exit(0);
    return 0;
}
