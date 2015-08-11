#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define LEARNING_RATE 0.6f
#define MAX_CYCLES 1000
#define MAX_ERROR 0.2f

typedef struct
{
	int weight_count;
	float *weights;
	float treshold;

} PERCEPTRON;


typedef struct
{
	int input_count;
	float *inputs;
	float result;
} PATTERN;

int evaluate(PERCEPTRON *per, PATTERN *pat)
{
	int i;
	float sum;

	assert(per->weight_count == pat->input_count);

	sum = 0;
	for (i = 0; i < per->weight_count; i++)
		sum += pat->inputs[i] * per->weights[i];

	sum += per->treshold;
	
	if (sum >= 0)
		return 1;
	return 0;
}

void train_aux(PERCEPTRON *per, PATTERN *pat)
{
	int i;
	int o;
	float delta;
	assert(per->weight_count == pat->input_count);

	o = evaluate(per, pat);
	delta = LEARNING_RATE * ( pat->result - o );
	for (i = 0; i < per->weight_count; i++)
	{
		per->weights[i] += delta * pat->inputs[i];
	}
	per->treshold += delta;
}

int train(PERCEPTRON *per, PATTERN *patterns, int pattern_count, int max_cycles, float max_error)
{
	int i, c;
	float error_value;
	float d, o;

	for (c = 0; c < max_cycles; c++)
	{
		for (i = 0; i < pattern_count; i++)
			train_aux(per, &patterns[i]);

		error_value = 0;
		/* compute error */
		for (i = 0; i < pattern_count; i++)
		{
			d = patterns[i].result;
			o = evaluate(per, &patterns[i]);
			error_value += 0.5 * ((d-o) * (d-o));
		}
		
		if (error_value < max_error)
			break;
	}

	return c;
}


void die(char *message)
{
	fprintf(stderr, "Chyba: %s\n", message);
	exit(1);
}

void perceptron_init(PERCEPTRON *per, int count)
{
	int i;

	per->weight_count = count;
	per->weights = (float *) malloc(sizeof(float) * count);
	if (per->weights == NULL)
		die("nedostatok pamate");

	for (i = 0; i < per->weight_count; i++)
		per->weights[i] = -1.0 + 2*((float)rand() / RAND_MAX);

	per->treshold = -1.0 + 2*((float)rand() / RAND_MAX);
}

void pattern_init(PATTERN *pat, int count)
{
	pat->input_count = count;
	pat->inputs = (float *) malloc(sizeof(float) * count);
	if (pat->inputs == NULL)
		die("nedostatok pamate");
}

int main(int argc, char *argv[])
{
	FILE *fp;
	int i, j;
	int cycles;

	int max_cycles;
	int pattern_len; /* x1, x2, ..., xn */
	int pattern_count; /* pocet vstupnych vzoriek */
	int input_count; /* pocet vstupnych vzoriek */
	PERCEPTRON *perceptrons;
	PATTERN *patterns;
	PATTERN *inputs;

	if (argc <= 2)
	{
		fprintf(stderr, "%s <training_file> <input_file> [<max_cyklov>]\n", argv[0]);
		return 0;
	}

	if (argc >= 4)
		max_cycles = atoi(argv[3]);
	else
		max_cycles = MAX_CYCLES;
	

	/* training data */
	fp = fopen(argv[1], "r");
	if (fp == NULL)
		die("Error opening training file");

	fscanf(fp, "%d %d", &pattern_len, &pattern_count);
	patterns = (PATTERN *) malloc(sizeof(PATTERN) * pattern_count);
	for (i = 0; i < pattern_count; i++)
	{
		pattern_init(&patterns[i], pattern_len);
		for (j = 0; j < pattern_len; j++)
			fscanf(fp, "%f", &patterns[i].inputs[j]);
		fscanf(fp, "%f", &patterns[i].result);
	}
	fclose(fp);

	/* input data */
	fp = fopen(argv[2], "r");
	if (fp == NULL)
		die("Error opening input file");

	fscanf(fp, "%d %d", &pattern_len, &input_count);
	inputs = (PATTERN *) malloc(sizeof(PATTERN) * input_count);
	for (i = 0; i < input_count; i++)
	{
		pattern_init(&inputs[i], pattern_len);
		for (j = 0; j < pattern_len; j++)
			fscanf(fp, "%f", &inputs[i].inputs[j]);
		fscanf(fp, "%f", &inputs[i].result);
	}
	fclose(fp);

	/* alocate one perceptron */
	perceptrons = (PERCEPTRON *) malloc(sizeof(PERCEPTRON) * 1);
	perceptron_init(&perceptrons[0], pattern_len);
	
	/* train network */
	cycles = train(&perceptrons[0], inputs, input_count, max_cycles, MAX_ERROR);
	printf("Trenovanie ukoncene po %d cykloch\n", cycles);

	printf("Perceptron:\n");
	for (i = 0; i < pattern_len; i++)
	{
		printf("%f ", perceptrons[0].weights[i]);
	}
	printf(" treshold: %f\n", perceptrons[0].treshold);

	printf("Testovanie vstupov:\n\t (ocakavana/vyhodnotena/rozdiel)\n");

	for (i = 0; i < input_count; i++)
	{
		float eval;
		eval = evaluate(&perceptrons[0], &inputs[i]);
		printf("\t%3d# \t %.0f\t %.0f\t [%.0f]\n", i+1, inputs[i].result, eval, inputs[i].result - eval);
	}

	return 0;
}

