struct node {
	struct node *left;
	struct node *right;
};

extern void process(struct node *);
void postorder_traverse(struct node *p)
{
	if (p->left)
#pragma omp task
		postorder_traverse(p->left);
	else if (p->right)
#pragma omp task
		postorder_traverse(p->right);

#pragma omp taskwait
	process(p);
}

