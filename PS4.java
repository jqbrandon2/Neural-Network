import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.io.FileWriter;


/**************************************************
Problem Set: Problem Set 4: Neural Network
Name: Brandon Evans
Synax: java PS4 w1.txt w2.txt xdata.txt ydata.txt
**************************************************/


public class PS4 extends Matrix {
	
	static double[][] X = new double[10000][785];
	static double[][] Y = new double[10000][1];
	static double[][] Y1 = new double[10000][10];
	static double[][] W1 = new double[30][785];
	static double[][] W2 = new double[10][30];
	static double[][] delta1 = new double [10000][30];
	static double[][] delta2 = new double [10000][10];
	static double[][] H1 = new double [10000][30];
	public static double accuracy;
	
	public static void main(String[] args) throws IOException {
		
		
		
		
		X = PS4.createInputMatrix("C://Users//encry//eclipse-workspace//AI//src//xdata.txt");
		W1 =PS4.createW1("C://Users//encry//eclipse-workspace//AI//src//w1.txt");
		W2 = PS4.createW2("C://Users//encry//eclipse-workspace//AI//src//w2.txt");
		Y = PS4.createY("C://Users//encry//eclipse-workspace//AI//src//ydata.txt");
		
		PS4.GradientDescent(X, Y, 0.001);
		
		
		
		
		
	}
	
	 public static void GradientDescent(double X[][], double Y[][], double alpha) throws IOException {
		 FileWriter writerloss = new FileWriter("loss.txt");
		 FileWriter writeraccuracy = new FileWriter("accuracy.txt");
		 
		 int epoch = 0;
		 int counter = 0;
		 
		 System.out.println(
		            "Training Phase:\n"+
		            	"--------------------------------------------------------------------------------------------------------\n" + "Number of Entries (n): " + X.length +"\n" + "Number of Features (p): " + 784
		        );
		 
		 System.out.println("");
		        System.out.println(
			            "Starting Gradient Descent:\n"+
			            "--------------------------------------------------------------------------------------------------------\n"
			        );
		 while(epoch<700) { 
					 H1 = calcHiddenLayer(X, W1);
					Y1 = calcOutputLayer(H1, W2);
					 double loss = calcLoss(Y, Y1, calcLambda(W1,W2), 10000);
					delta2 = delta2(Y1, Y);
					delta1 = delta1(delta2, W2, X, W1);
					 
					W2 = updateW2(W2, deltaW2(delta2,H1));
					W1 = updateW1(W1, deltaW1(delta1, X));
					 epoch++;
					 writerloss.write(epoch +", "+loss +"\n");
					 writeraccuracy.write(epoch +", "+accuracy +"\n");
					 System.out.println("Epoch " +epoch + " :" + "\t" + "Loss of " + loss + "\t" + "Delta = " + accuracy + "%" + "\t" + "Epsilon = " + accuracy + "%");
					 
				 }
		 
		 		System.out.println("");
		 		System.out.println("Epochs Required:" +"\t" + epoch);
		 		
		 		writerloss.close();
		 		writeraccuracy.close();
			
			 System.out.println("Forward and Backwards Pass is complete.");
			 
			 FileWriter writerw1 = new FileWriter("w1out.txt");
			 FileWriter writerw2 = new FileWriter("w2out.txt");
			 
			 for(int i = 0; i<30; i++) {
				     writerw1.write("\n");
				 for(int j = 0; j<785; j++) {
					 writerw1.write((int) W1[i][j] +", ");
				 }
			 } writerw1.close();
			 System.out.println("Finished writing new W1 weights");
			 
			 for(int i = 0; i<10; i++) {
			     writerw2.write("\n");
			 for(int j = 0; j<30; j++) {
				 writerw2.write((int) W1[i][j] +", ");
			 	}
			 } writerw2.close();
			 System.out.println("Finished writing new W2 weights");
		 }
		 
		 
	 
	 public static double sigmoid(double x) {
		 return 1/( 1 + Math.exp(-x) );
	 }
	
	 public static double dsigmoid(double x) {
		return sigmoid(x) * (1- sigmoid(x));
		 
	 }
	 
	 public static double[][] calcHiddenLayer(double X[][], double W1[][]) {
		double temp[][] = multiply(X, transpose(W1));
		double[][] H1 = new double [10000][30]; 
		
		for(int i = 0; i<10000; i++) {
			for(int j = 0; j<30; j++) {
				H1[i][j] = sigmoid(temp[i][j]);
			}
		}
		return H1;
	 }
	 
	 public static double[][] calcOutputLayer(double H1[][], double W2[][]) {
			double temp[][] = multiply(H1, transpose(W2));
			double[][] Y1 = new double [10000][10]; 
			
			for(int i = 0; i<10000; i++) {
				for(int j = 0; j<10; j++) {
					Y1[i][j] = sigmoid(temp[i][j]);
				}
			}
			return Y1;
		 }
	 
	 public static double calcLambda(double W1[][], double W2[][]) {
		 double sum = 0;
		 int count = 0;
		 for(int i = 0; i<30; i++) {
			for(int j = 0; j<785; j++) {
				double weight = W1[i][j];
				sum += Math.pow(weight, 2);
				count++;
			}
		}
		 
		 double sum1 = 0;
		 int count1 = 0;
		 for(int k = 0; k<10; k++) {
			 for(int l = 0; l<30; l++) {
				 double weight1 = W2[k][l];
				 sum1+= Math.pow(weight1, 2);
				 count1++;
			 }
		 }
		
		double totalSum = sum + sum1;
		double totalCount = count + count1;
		
		return totalSum/totalCount;
		 
	 }
	 
	 public static double calcLoss(double Y[][], double Y1[][], double lambda, int n) {
		 double loss = 0;
		 for(int i = 0; i<10000; i++) {
			 for(int k = 0; k <10; k++) {
				 loss += (1/n * (-Y[i][0] * Math.log(Y1[i][k]) - (1 - Y[i][0]) * Math.log(1 - Y1[i][k])) + lambda);
			 }
		 }
		 return loss;
	 }
	 
	 public static double[][] delta2(double Y1[][], double Y[][]) {
		 double[][] delta2 = new double [10000][10];
		 for(int i = 0; i<10000; i++) {
			 for(int j = 0; j<10; j++) {
				 delta2[i][j] = Y1[i][j] - Y[i][0];
				 accuracy = delta2[i][j];
			 }
		 }
		return delta2;
	 }
	 
	 public static double[][] delta1(double delta2[][], double W2[][], double X[][], double W1 [][]) {
		 double[][] delta1 = new double [10000][30];
		 double temp1[][] = multiply(delta2, W2);
		 double temp2[][] = multiply(X, transpose(W1));
		 
		 
		 for(int i = 0; i<10000; i++) {
			 for(int j = 0; j<30; j++) {
				  delta1[i][j] = temp1[i][j] * dsigmoid(temp2[i][j]);
			 }
		 }
		return delta1;
	 }
	 
	 public static double[][] deltaW2(double delta2[][], double H1[][]) {
		 return multiply(transpose(delta2), H1); 
	 }
	 
	 public static double[][] deltaW1(double delta1[][], double X[][]) {
		 return multiply(transpose(delta1), X); 
	 }
	 
	 public static double[][] updateW2(double W2[][], double deltaW2[][]) {
		 for(int i = 0; i<10; i++) {
			 for(int j = 0; j<30; j++) {
				 W2[i][j] = W2[i][j] - 0.25 * deltaW2[i][j];
			 }
		 }
		return deltaW2;
	 }
	 
	 public static double[][] updateW1(double W1[][], double deltaW1[][]) {
		 for(int i = 0; i<30; i++) {
			 for(int j = 0; j<785; j++) {
				 W1[i][j] = W1[i][j] - 0.25 * deltaW1[i][j];
			 }
		 }
		return deltaW1;
	 }
	 
	 
	 
	 
	 
	 
	 
	 
	 public static double[][] createInputMatrix(String file) throws NumberFormatException, IOException {
		 
		 int count = 0;
		 BufferedReader br = new BufferedReader(new FileReader(file));
		 String line;
		 double[][] X = new double [10000][785];
			
			while ((line = br.readLine()) != null) {
				count++;
				String[] value = line.split(",");
				
				for(int i = 0; i<10000; i++) {
					for(int j = 1; j< 784; j++) {
						X[i][j] = Double.parseDouble(value[j]);
						X[i][0] = 1.0;
				}
						
					
				}
				System.out.println(count);
			}
			
			br.close();
			System.out.println("Done creating X matrix.");
			return X;
	 }
	 
	 public static double[][] createW1(String file) throws NumberFormatException, IOException {
		 
		 
		 BufferedReader br = new BufferedReader(new FileReader(file));
		 String line;
		 double[][] W1 = new double [30][785];
			while ((line = br.readLine()) != null) {
				String[] value = line.split(",");
				for(int i = 0; i<30; i++) {
					for(int j = 0; j< 785; j++) {
						W1[i][j] = Double.parseDouble(value[j]);		
				}	
					
				}
			}
			
			br.close();
			System.out.println("Done creating W1 matrix.");
			return W1;
	 }
	 
	 public static double[][] createW2(String file) throws NumberFormatException, IOException {
		 
		 
		 BufferedReader br = new BufferedReader(new FileReader(file));
		 String line;
		 double[][] W2 = new double [10][30];
			
			while ((line = br.readLine()) != null) {
				String[] value = line.split(",");
				for(int i = 0; i<10; i++) {
					for(int j = 0; j< 30; j++) {
						W2[i][j] = Double.parseDouble(value[j]);		
				}
						
					
				}
				
			}
			
			br.close();
			System.out.println("Done creating W2 matrix.");
			return W2;
	 }
	 
	 public static double[][] createY(String file) throws NumberFormatException, IOException {
		 
		 
		 BufferedReader br = new BufferedReader(new FileReader(file));
		 String line;
		 double[][] Y = new double [10000][1];
			
			while ((line = br.readLine()) != null) {
				String[] value = line.split("\n");
				
				for(int i = 0; i<10000; i++) {
					for(int j = 0; j<1; j++) {
						
						Y[i][j] = Double.parseDouble(value[j]);		
					}
				}	
				
			}		
				
			
			
			br.close();
			System.out.println("Done creating Y matrix.");
			return Y;
	 }
	 }

