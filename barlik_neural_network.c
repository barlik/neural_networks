/*
 * 2007 Rastislav Barlik
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>

#ifndef PATH_MAX
#define PATH_MAX 255
#endif


double *etrain;
double *etest;

int *strain;
int *stest;

int *cycles_count;

int train_count;
int test_count;

int test_index = 0;

struct
{
	double learning_rate;
	double momentum;
	int max_cycles;
	double max_error;
	int train_test_ratio;
	int hidden_count;
	int delete_weights;
	char config_file[PATH_MAX+1];
	char training_file[PATH_MAX+1];
	char input_file[PATH_MAX+1];
	char output_file[PATH_MAX+1];

} config = { 0.6f, 0.7f, 1000, 0.02f, 70, 20, 20, "config.txt", "xor.txt", "input_xor.txt", "output.txt" };

typedef struct
{
	int layer;
	int weight_count;
	double *weights;
	int *weight_deleted;
	double output;
	double der;
	double err;
	double dw;
	double treshold;
	int treshold_deleted;
} NEURAL;


int NEURAL_OUTPUT_INDEX;

typedef struct
{
	int train;
	int input_count;
	double *inputs;
	double result;
} PATTERN;

void reset_counts()
{
	int i;

	for (i = 0; i < config.max_cycles; i++)
	{
		etrain[i] = 0;
		etest[i] = 0;
		strain[i] = 0;
		stest[i] = 0;
	}
}

void die(char *message)
{
	fprintf(stderr, "Chyba: %s\n", message);
	exit(1);
}

void read_config(char *filename)
{
	FILE *fp;
	char line[1024], *s;

	fp = fopen(filename, "r");
	if (fp == NULL)
		die("citanie konfigu zlyhalo");

	while (fgets(line, sizeof(line), fp) != NULL)
	{
		line[strlen(line)-1] = '\0';

		s = strtok(line, " ");
		if (!s)
			die("nespravny format konfigu");
		if (!strcasecmp(s, "learning_rate")) { s = strtok(NULL, " "); config.learning_rate = strtod(s, NULL);}
		else if (!strcasecmp(s, "momentum")) { s = strtok(NULL, " "); config.momentum = strtod(s, NULL); }
		else if (!strcasecmp(s, "max_error")) { s = strtok(NULL, " "); config.max_error = strtod(s, NULL); }
		else if (!strcasecmp(s, "max_cycles")) { s = strtok(NULL, " "); config.max_cycles = atoi(s); }
		else if (!strcasecmp(s, "train_test_ratio")) { s = strtok(NULL, " "); config.train_test_ratio = atoi(s); }
		else if (!strcasecmp(s, "hidden_count")) { s = strtok(NULL, " "); config.hidden_count = atoi(s); }
		else if (!strcasecmp(s, "delete_weights")) { s = strtok(NULL, " "); config.delete_weights = atoi(s); }
		else if (!strcasecmp(s, "training_file")) { s = strtok(NULL, " "); strcpy(config.training_file, s); }
		else if (!strcasecmp(s, "input_file")) { s = strtok(NULL, " "); strcpy(config.input_file, s); }
		else if (!strcasecmp(s, "output_file")) { s = strtok(NULL, " "); strcpy(config.output_file, s); }
	}
	fclose(fp);
}

void shuffle(PATTERN *patterns, int pattern_count)
{
	int i;
	int r1, r2;
	PATTERN temp;

	for (i = 0; i < pattern_count/2; i++)
	{
		r1 = rand()%pattern_count;
		do {
			r2 = rand()%pattern_count;
		} while (r2 == r1);
		
		memcpy(&temp, &patterns[r1], sizeof(PATTERN));
		memcpy(&patterns[r1], &patterns[r2], sizeof(PATTERN));
		memcpy(&patterns[r2], &temp, sizeof(PATTERN));
	}
}

void delete_weights(NEURAL *neurals, int neural_count)
{
	int c, i, r, j, weight_count = 0, delete;
	int *w;

	for (i = 0; i < neural_count; i++)
		weight_count += neurals[i].weight_count + 1;
	
	delete = weight_count * config.delete_weights / 100.0;
	printf("Mazem %d%% synapsii\n", config.delete_weights);

	w = (int *) malloc(sizeof(int) * weight_count);
	memset(w, 0, weight_count);

	for (i = 0; i < delete; i++)
	{
		do {
			r = rand()%weight_count;
		} while (w[r]);

		w[r] = 1;
	}

	c = 0;
	for (i = 0; i < neural_count; i++)
	{
		for (j = 0; j < neurals[i].weight_count; j++)
		{
			if (w[c])
				neurals[i].weight_deleted[j] = 1;
			c++;
		}
		if (w[c])
			neurals[i].treshold_deleted = 1;
		c++;
	}

	free(w);
}


double activation_func(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double evaluate_aux(NEURAL *neurals, int neural_count, int actual)
{
	int i, j, v;
	double sum;

	sum = 0;
	i = 0;
	for (j = 0; j < neural_count; j++)
	{
		if (neurals[actual].layer-1 != neurals[j].layer)
			continue;
		if (neurals[actual].weight_deleted[i] == 0)
			sum += neurals[j].output * neurals[actual].weights[i];
		i++;
	}
	if (neurals[actual].treshold_deleted == 0)
		sum += neurals[actual].treshold;

	return sum;
}

int classify(PATTERN *patterns, int pattern_count, double o)
{
	int i;
	int nearest;

	nearest = 0;
	for (i = 1; i < pattern_count; i++)
	{
		if (fabs(patterns[i].result - o) < fabs(patterns[nearest].result - o))
			nearest = i;
	}

	return nearest;
}

int classified(PATTERN *patterns, int pattern_count, double d, double o)
{
	int i;
	int nearest;

	nearest = 0;
	for (i = 1; i < pattern_count; i++)
	{
		if (fabs(patterns[i].result - o) < fabs(patterns[nearest].result - o))
			nearest = i;
	}

	if (fabs(patterns[nearest].result - d) < 0.005)
		return 1;

	return 0;
}

double forward(NEURAL *neurals, int neural_count, PATTERN *in)
{
	int i, v;
	int layer = 0;
	int found;

	v = 0;
	for (i = 0; i < neural_count; i++)
	{
		if (neurals[i].layer == 0) {
			neurals[i].output = neurals[i].weights[0] * in->inputs[i];
		} else {
			neurals[i].output = activation_func(evaluate_aux(neurals, neural_count, i));
			neurals[i].der = neurals[i].output * (1 - neurals[i].output); 
		}
	}

	return neurals[neural_count-1].output;
}

int backward(NEURAL *neurals, int neural_count, PATTERN *pattern)
{
	int i, k, v;

	for (i = 0; i < neural_count; i++)
		neurals[i].err = 0;

	for (i = neural_count -1; i > 0; i--)
	{
		if (neurals[i].layer == 0)
			break;
		/* output neural */
		if (neurals[i].layer == neurals[neural_count-1].layer)
		{
			neurals[i].err += pattern->result - neurals[i].output;
		}

		neurals[i].err = neurals[i].err * neurals[i].der;
		
		/* vchadzajuce neurony */
		v = 0;
		for (k = 0; k < neural_count; k++)
		{
			if (neurals[k].layer != neurals[i].layer-1)
				continue;
			if (neurals[k].layer == 0)
				continue;

			neurals[k].err += neurals[i].weights[v] * neurals[i].err;
			v++;
		}
	}
}

int modify_weights(NEURAL *neurals, int neural_count)
{
	int i, v, k;

	/* zmena vah */
	for (i = 0; i < neural_count; i++)
	{
		if (neurals[i].layer == 0)
			continue;
		v = 0;
		for (k = 0; k < neural_count; k++)
		{
			if (neurals[i].layer - 1 != neurals[k].layer)
				continue;
			neurals[i].dw = config.momentum * neurals[i].dw + config.learning_rate * neurals[i].err * neurals[k].output;
			neurals[i].weights[v] += neurals[i].dw;
			v++;
		}
		neurals[i].treshold += config.learning_rate * neurals[i].err;
	}
}

int train(NEURAL *neurals, int neural_count, PATTERN *patterns, int pattern_count, int learn)
{
	int i, c, j, k, v, v2, v3;
	double error_value, sum;
	int trainp, testp;
	double e, d, o, deltaW, deltaH;

	reset_counts();
	for (i = 0; i < neural_count; i++)
		neurals[i].dw = 0;

	for (c = 0; c < config.max_cycles; c++)
	{
		for (j = 0; j < pattern_count; j++)
		{
			if (patterns[j].train == 0)
				continue;

			forward(neurals, neural_count, &patterns[j]);
			backward(neurals, neural_count, &patterns[j]);
			if (learn)
				modify_weights(neurals, neural_count);
		}

		error_value = 0.0f;
		/* compute error */
		for (i = 0; i < pattern_count; i++)
		{
			//evaluate(neurals, neural_count, &patterns[i]);
			o = forward(neurals, neural_count, &patterns[i]);
			d = patterns[i].result;
			//printf("%f %f\n", d, o);
			error_value += 0.5 * ((d-o) * (d-o));

			if (patterns[i].train == 1)
			{
				etrain[c] += 0.5 * ((d-o) * (d-o));
				strain[c] += classified(patterns, pattern_count, d, o);
			}
			else
			{
				etest[c] += 0.5 * ((d-o) * (d-o));
				stest[c] += classified(patterns, pattern_count, d, o);
			}
		}
#if 1
		if (train_count)
			trainp = strain[c]*100/train_count;
		else
			trainp = 0;
		if (test_count)
			testp = stest[c]*100/test_count;
		else
			testp = 0;

		printf("%d %f %f %d %d\n", c, etrain[c], etest[c], trainp, testp);
#endif
		
		if (error_value < config.max_error)
			return c;

		shuffle(patterns, pattern_count);
		if (!learn)
			return c + 1;
	}

	return c;
}

void neural_init(NEURAL *per, int layer, int weight_count, int random_fill)
{
	int i;

	per->layer = layer;
	per->weight_count = weight_count;
	per->weights = (double *) malloc(sizeof(double) * weight_count);
	per->weight_deleted = (int *) malloc(sizeof(int) * weight_count);
	if (per->weights == NULL)
		die("nedostatok pamate");

	for (i = 0; i < per->weight_count; i++)
		per->weight_deleted[i] = 0;
	if (random_fill)
	{
		for (i = 0; i < per->weight_count; i++)
			per->weights[i] = -1.0 + 2*((double)rand() / RAND_MAX);
	}
}

void random_weights(NEURAL *neurals, int neural_count)
{
	int i, j;

	for (j = 0; j < neural_count; j++)
	for (i = 0; i < neurals[i].weight_count; i++)
		neurals[j].weights[i] = -1.0 + 2*((double)rand() / RAND_MAX);
}

void pattern_init(PATTERN *pat, int count)
{
	pat->input_count = count;
	pat->inputs = (double *) malloc(sizeof(double) * count);
	pat->train = 0;
	if (pat->inputs == NULL)
		die("nedostatok pamate");
}

double cubic_error(NEURAL *neurals, int neural_count, PATTERN *patterns, int pattern_count)
{
	double error = 0.0f;
	double eval;
	int i;

	for (i = 0; i < pattern_count; i++)
	{
		eval = forward(neurals, neural_count, &patterns[i]);
		error += 0.5 * (patterns[i].result - eval) * (patterns[i].result - eval);
	}
	return error;
}

/* kolko stlpcov mame v subore */
int columns(char *str)
{
	int ret = 0;
	char *s;


	s = strtok(str, "| \t");
	if (s == NULL)
		return 0;

	while (s)
	{
		s = strtok(NULL, "| \t");
		ret++;
	}

	return ret;
}

void split_patterns(PATTERN *patterns, int pattern_count)
{
	int i, p = config.train_test_ratio * pattern_count / 100;

	for (i = 0; i < p; i++)
		patterns[i].train = 1;

	train_count = p;
	test_count = pattern_count-p;
	printf("Pomer tren/test vzoriek je %d:%d\n", train_count, test_count);
}

int main(int argc, char *argv[])
{
	FILE *fp;
	char line[1024], *s;
	int i, j, layer;
	int cycles;

	int max_cycles;
	int pattern_len; /* x1, x2, ..., xn */
	int pattern_count; /* pocet vstupnych vzoriek */
	double *pattern_max; /* maxima atributov vzoriek */
	double result_max; /* maxima vystupu vzoriek */
	double error;
	int input_count; /* pocet vstupnych vzoriek */
	NEURAL *neurals;
	int neural_count;
	PATTERN *patterns;
	PATTERN *inputs;
	int inp_count;

	srand(time(0));

	if (argc > 1)
		read_config(argv[1]);
	else
		read_config(config.config_file);

	/* training data */
	fp = fopen(config.training_file, "r");
	if (fp == NULL)
		die("otvarani trenovacieho suboru");

	if (fgets(line, sizeof(line), fp) == NULL)
		die("trenovaci subor je prazdny");
	line[strlen(line)-1] = '\0';

	pattern_len = columns(line) - 1;
	if (pattern_len <= 0)
		die("nespravny format vstupu");

	pattern_max = (double *) malloc(sizeof(double) * pattern_len);

	pattern_count = 1;
	while (fgets(line, sizeof(line), fp) != NULL)
		pattern_count++;
	rewind(fp); /* fseek(fp, 0L, SEEK_SET); */

	printf("pocet vstupnych vzoriek: %d\n", pattern_count);
	printf("Pocet atributov: %d\n", pattern_len);

	patterns = (PATTERN *) malloc(sizeof(PATTERN) * pattern_count);

	etrain = (double *)malloc(sizeof(double) * (config.max_cycles));
	etest = (double *)malloc(sizeof(double) * (config.max_cycles));
	strain = (int *)malloc(sizeof(int) * (config.max_cycles));
	stest = (int *)malloc(sizeof(int) * (config.max_cycles));

	for (i = 0; i < pattern_count; i++)
	{
		if (fgets(line, sizeof(line), fp) == NULL)
			die("chyba");

		line[strlen(line)-1]='\0';

		s = strtok(line, "| \t");
		if (s == NULL)
			die("nespravny format suboru");

		pattern_init(&patterns[i], pattern_len);
		for (j = 0; j < pattern_len; j++)
		{
			patterns[i].inputs[j] = strtod(s, NULL);
			s = strtok(NULL, "| \t");
		}
		patterns[i].result = strtod(s, NULL);

		/* check max values */
		for (j = 0; j < pattern_len; j++)
		{
			if (patterns[i].inputs[j] > pattern_max[j])
				pattern_max[j] = patterns[i].inputs[j];
		}

		if (patterns[i].result > result_max)
			result_max = patterns[i].result;
	}
	fclose(fp);

	/* spracovanie vstupov */
	fp = fopen(config.input_file, "r");
	if (fp == NULL)
		die("otvarani trenovacieho suboru");
	inp_count = 0;
	while (fgets(line, sizeof(line), fp) != NULL)
		inp_count++;
	rewind(fp); /* fseek(fp, 0L, SEEK_SET); */

	inputs = (PATTERN *) malloc(sizeof(PATTERN) * inp_count);
	for (i = 0; i < inp_count; i++)
	{
		if (fgets(line, sizeof(line), fp) == NULL)
			die("chyba");

		line[strlen(line)-1]='\0';

		s = strtok(line, " \t");
		if (s == NULL)
			die("nespravny format suboru");

		pattern_init(&inputs[i], pattern_len);
		for (j = 0; j < pattern_len; j++)
		{
			inputs[i].inputs[j] = strtod(s, NULL);
			s = strtok(NULL, " \t");
		}
	}
	fclose(fp);

	/* !spracovanie vstupov */
#if 0
	for (i = 0; i < pattern_count; i++)
	{
		patterns[i].result = patterns[i].result / result_max;
		printf("%d ", i);
		for (j = 0; j < pattern_len; j++)
		printf("%lf ", patterns[i].inputs[j]);
		printf("| %lf ", patterns[i].result);
		printf("\n");
	}
#endif

	/* shuffle and split patterns */
	shuffle(patterns, pattern_count);
	split_patterns(patterns, pattern_count);

	/* alocate neurals */
	neural_count = pattern_len + config.hidden_count + 1;
	neurals = (NEURAL *) malloc(sizeof(NEURAL) * (neural_count));

	/* initialize input neurons */
	layer = 0;
	for (i = 0; i < pattern_len; i++)
	{
		neural_init(&neurals[i], 0, layer, 0);
		neurals[i].weights[0] = 1.0f / pattern_max[i];
	}
	layer++;
#if 0
	for (i = 0; i < pattern_count; i++)
	for (j = 0; j < pattern_len; j++)
		printf("%f ", patterns[i].inputs[j] * neurals[j].weights[0]);
#endif
	/* initialize hidden neurons */
	for (i = 0; i < config.hidden_count; i++)
	{
		neural_init(&neurals[pattern_len + i], layer, pattern_len, 1);
	}
	if (config.hidden_count)
		layer++;

	/* initialize output neuron */
	NEURAL_OUTPUT_INDEX = pattern_len+config.hidden_count;
	neural_init(&neurals[NEURAL_OUTPUT_INDEX], layer, config.hidden_count, 1);
	
#if 0
	evaluate(neurals, neural_count, &inputs[0]);
	printf("%f\n", neurals[NEURAL_OUTPUT_INDEX].output);
	for (i = 0; i < neurals[NEURAL_OUTPUT_INDEX].weight_count; i++)
		printf("%d %f\n", neurals[NEURAL_OUTPUT_INDEX].layer, neurals[NEURAL_OUTPUT_INDEX].weights[i]);
#endif
	/* train network */
	printf("Rychlost ucenia: %f\n", config.learning_rate);
	printf("Max cyklov: %d\n", config.max_cycles);
	printf("Max chyba: %f\n", config.max_error);
	printf("Trenujem\n");
	cycles = train(neurals, neural_count, patterns, pattern_count, 1);

	if (cycles < config.max_cycles)
		printf("Trenovanie ukoncene po %d cykloch\n", cycles);
	else
		printf("Dosiahnuty maximalny pocet cyklov\n");

	error = cubic_error(neurals, neural_count, patterns, pattern_count);
	printf("Kvadraticka chyba: %f\n", error);
	printf("Percento spravne klasifikovanych vstupov: %d %d %d\n", strain[cycles-1],stest[cycles-1],  pattern_count);
	printf("Percento spravne klasifikovanych vstupov: %d%%\n", (strain[cycles-1]+stest[cycles-1])*100 / pattern_count);

	if (config.delete_weights)
	{
		delete_weights(neurals, neural_count);
		cycles = train(neurals, neural_count, patterns, pattern_count, 0);
		printf("%d\n", cycles);
		printf("Percento spravne klasifikovanych vstupov: %d%%\n", (strain[cycles-1]+stest[cycles-1])*100 / pattern_count);
		printf("Dotrenovanie siete\n");

		cycles = train(neurals, neural_count, patterns, pattern_count, 1);
		error = cubic_error(neurals, neural_count, patterns, pattern_count);
		printf("Kvadraticka chyba: %f\n", error);
		printf("Percento spravne klasifikovanych vstupov: %d%%\n", (strain[cycles-1]+stest[cycles-1])*100 / pattern_count);
	}
	
	/* evaluate input file and print results to output file */
	fp = fopen(config.output_file, "w");
	if (fp == NULL)
		die("otvarani vystupneho suboru");
	for (i = 0; i < inp_count; i++)
	{
		double eval;
		for (j = 0; j < inputs[i].input_count; j++)
		fprintf(fp, "%.2f\t", inputs[i].inputs[j]); 
		eval = forward(neurals, neural_count, &inputs[i]) * result_max;
		fprintf(fp, "|\t");
		fprintf(fp, "%.02f\n", patterns[classify(patterns, pattern_count, eval)].result);
	}

	fclose(fp);

	return 0;
}
