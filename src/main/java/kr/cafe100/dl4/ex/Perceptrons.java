package kr.cafe100.dl4.ex;

import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static DLWJ.util.ActivationFunction.step;
import DLWJ.util.*;

/**
 * P196
 * Perceptrons
 * @author hohoonlee
 */
public class Perceptrons {
    public int nIn;     //입력 데이터의 차원(행)
    public INDArray w;
    
    public Perceptrons(int nIn) {
        this.nIn = nIn;
        w = Nd4j.create(new double[nIn], new int[]{nIn, 1});
    }
    
    public int train(INDArray x, INDArray t, double learningRate) {

        int classified = 0;

        // DATA가 제대로 분류되었는지 확인
        double c = x.mmul(w).getDouble(0) * t.getDouble(0);

        // DATA가 틀리게 분류되었다면 최급 기울기 하강 방법을 적용한다
        if (c > 0) {
            classified = 1;
        } else {
            w.addi(x.transpose().mul(t).mul(learningRate));
        }

        return classified;
    }

    public int predict(INDArray x) {

        return step(x.mmul(w).getDouble(0));
    }

    
    public static void main(String[] argv) {
        //
        // Declare (Prepare) variables and constants for perceptrons
        //

        final int train_N = 1000;   //훈련용 DATA 수
        final int test_N = 200;     //시험용 DATA 수
        final int nIn = 2;          // dimensions of input data

        //학습용 입력 DATA
        INDArray train_X = Nd4j.create(new double[train_N * nIn], new int[]{train_N, nIn});
        //라벨링된 학습용 출력 DATA
        INDArray train_T = Nd4j.create(new double[train_N], new int[]{train_N, 1});

        //시험용 입력 DATA
        INDArray test_X = Nd4j.create(new double[test_N * nIn], new int[]{test_N, nIn});
        //라벨링된 시험용 출력 DATA
        INDArray test_T = Nd4j.create(new double[test_N], new int[]{test_N, 1});
        
        //모델이 예측한 출력 DATA
        INDArray predicted_T = Nd4j.create(new double[test_N], new int[]{test_N, 1});


        final int epochs = 2000;   // maximum training epochs
        final double learningRate = 1.;  // learning rate can be 1 in perceptrons


        //
        // Create training data and test data for demo.
        //
        // Let training data set for each class follow Normal (Gaussian) distribution here:
        //   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
        //   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
        //

        final Random rng = new Random(1234);  // seed random
        GaussianDistribution g1 = new GaussianDistribution(-2.0, 1.0, rng);
        GaussianDistribution g2 = new GaussianDistribution(2.0, 1.0, rng);


        // data set in class 1
        for (int i = 0; i < train_N/2 - 1; i++) {
            train_X.put(i, 0, Nd4j.scalar(g1.random()));
            train_X.put(i, 1, Nd4j.scalar(g2.random()));
            train_T.put(i, Nd4j.scalar(1));
        }
        for (int i = 0; i < test_N/2 - 1; i++) {
            test_X.put(i, 0, Nd4j.scalar(g1.random()));
            test_X.put(i, 1, Nd4j.scalar(g2.random()));
            test_T.put(i, Nd4j.scalar(1));
        }

        // data set in class 2
        for (int i = train_N/2; i < train_N; i++) {
            train_X.put(i, 0, Nd4j.scalar(g2.random()));
            train_X.put(i, 1, Nd4j.scalar(g1.random()));
            train_T.put(i, Nd4j.scalar(-1));
        }
        for (int i = test_N/2; i < test_N; i++) {
            test_X.put(i, 0, Nd4j.scalar(g2.random()));
            test_X.put(i, 1, Nd4j.scalar(g1.random()));
            test_T.put(i, Nd4j.scalar(-1));
        }


        //
        // Build SingleLayerNeuralNetworks model
        //

        int epoch = 0;  // training epochs

        // 퍼셉트론 구축
        Perceptrons classifier = new Perceptrons(nIn);

        // 학습 모델
        while (true) {
            int classified_ = 0;

            for (int i=0; i < train_N; i++) {
                classified_ += classifier.train(train_X.getRow(i), train_T.getRow(i), learningRate);
            }

            if (classified_ == train_N) break;  //모든 DATA가 분류 되었을 때

            epoch++;
            if (epoch > epochs) break;
        }


        // test
        for (int i = 0; i < test_N; i++) {
            predicted_T.put(i, Nd4j.scalar(classifier.predict(test_X.getRow(i))));
        }


        //
        // Evaluate the model
        //

        int[][] confusionMatrix = new int[2][2];
        double accuracy = 0.;
        double precision = 0.;
        double recall = 0.;

        for (int i = 0; i < test_N; i++) {

            if (predicted_T.getRow(i).getDouble(0) > 0) {
                if (test_T.getRow(i).getDouble(0) > 0) {
                    accuracy += 1;
                    precision += 1;
                    recall += 1;
                    confusionMatrix[0][0] += 1;
                } else {
                    confusionMatrix[1][0] += 1;
                }
            } else {
                if (test_T.getRow(i).getDouble(0) > 0) {
                    confusionMatrix[0][1] += 1;
                } else {
                    accuracy += 1;
                    confusionMatrix[1][1] += 1;
                }
            }

        }

        accuracy /= test_N;
        precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
        recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

        System.out.println("----------------------------");
        System.out.println("Perceptrons model evaluation");
        System.out.println("----------------------------");
        System.out.printf("Accuracy:  %.1f %%\n", accuracy * 100);
        System.out.printf("Precision: %.1f %%\n", precision * 100);
        System.out.printf("Recall:    %.1f %%\n", recall * 100);
    }
}
