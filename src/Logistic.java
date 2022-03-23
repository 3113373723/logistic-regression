import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

import static java.lang.Math.abs;

/**
 * Performs simple logistic regression.
 * User: tpeng
 * Date: 6/22/12
 * Time: 11:01 PM
 * 
 * @author tpeng
 * @author Matthieu Labas
 */
public class Logistic {

	/** the learning rate */
	private double rate;

	/** the weight to learn */
	private double[] weights;

	/** the number of iterations */
	private int ITERATIONS = 3000;

	/** 梯度下降的绝对值大小*/
	private double[] laseGradient;
	private double avgGradient;
	private double sigma = 0.1;

	public Logistic(int n, int num) {
		this.rate = 0.0001;
		weights = new double[n];
		laseGradient = new double[num];
	}

	private static double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	public void train(List<Instance> instances) throws IOException {
		SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
		Date date = new Date(System.currentTimeMillis());
		String output = "E:\\graduationProject\\logistic-regression\\out\\Log-" + formatter.format(date) + ".txt";
		File file = new File(output);
		FileWriter fw = new FileWriter(file);
		BufferedWriter bw = new BufferedWriter(fw);

		/** 获取当前系统时间*/
		long startTime = System.currentTimeMillis();
		int n = 0;
		while(n<ITERATIONS) {
			double lik = 0.0;
			n += 1;
			for (int i=0; i<instances.size(); i++) {
				int[] x = instances.get(i).x;
				double predicted = classify(x);
				int label = instances.get(i).label;
				laseGradient[i] = 0.0;
				for (int j=0; j<weights.length; j++) {
					weights[j] = weights[j] + rate * (label - predicted) * x[j];
					laseGradient[i]+=abs((label - predicted) * x[j]);
				}
				// not necessary for learning
				lik += label * Math.log(classify(x)) + (1-label) * Math.log(1- classify(x));
			}
			System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
			bw.write("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
			bw.newLine();

			// lazyTrain
//			lik = 0.0;
			n += 1;
			avgGradient = 0.0;
			for(double gradient:laseGradient){
				avgGradient+=gradient;
			}
			avgGradient = avgGradient/laseGradient.length;
			for (int i=0; i<instances.size(); i++) {
				if(laseGradient[i]<sigma*avgGradient)continue;
				int[] x = instances.get(i).x;
				double predicted = classify(x);
				int label = instances.get(i).label;
				for (int j=0; j<weights.length; j++) {
					weights[j] = weights[j] + rate * (label - predicted) * x[j];
				}
				// not necessary for learning
//				lik += label * Math.log(classify(x)) + (1-label) * Math.log(1- classify(x));
			}
//			System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
//			bw.write("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + lik);
//			bw.newLine();

		}
		/** 获取当前的系统时间，与初始时间相减就是程序运行的毫秒数，除以1000就是秒数*/
		long endTime = System.currentTimeMillis();
		double usedTime = (endTime - startTime) / (double) 1000;
		bw.write(" Train Time is : " + usedTime + "s");
		bw.close();
		fw.close();
	}

	private double classify(int[] x) {
		double logit = .0;
		for (int i=0; i<weights.length;i++)  {
			logit += weights[i] * x[i];
		}
		return sigmoid(logit);
	}

	public static class Instance {
		public int label;
		public int[] x;

		public Instance(int label, int[] x) {
			this.label = label;
			this.x = x;
		}
	}

	public static List<Instance> readDataSet(String file) throws FileNotFoundException {
		List<Instance> dataset = new ArrayList<Instance>();
		Scanner scanner = null;
		try {
			scanner = new Scanner(new File(file));
			while(scanner.hasNextLine()) {
				String line = scanner.nextLine();
				if (line.startsWith("#")) {
					continue;
				}
				String[] columns = line.split("\\s+");

				// skip first column and last column is the label
				int i = 1;
				int[] data = new int[columns.length-2];
				for (i=1; i<columns.length-1; i++) {
					data[i-1] = Integer.parseInt(columns[i]);
				}
				int label = Integer.parseInt(columns[i]);
				Instance instance = new Instance(label, data);
				dataset.add(instance);
			}
		} finally {
			if (scanner != null)
				scanner.close();
		}
		return dataset;
	}


	public static void main(String... args) throws IOException {
		List<Instance> instances = readDataSet("dataset.txt");
		Logistic logistic = new Logistic(5,instances.size());
		logistic.train(instances);
		int[] x = {2, 1, 1, 0, 1};
		System.out.println("prob(1|x) = " + logistic.classify(x));

		int[] x2 = {1, 0, 1, 0, 0};
		System.out.println("prob(1|x2) = " + logistic.classify(x2));

		int[] x3 = {1, 1, 0, 0, 0};
		System.out.println("prob(1|x3) = " + logistic.classify(x3));
	}

}
