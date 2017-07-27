package org.deeplearning4j.examples.deepbelief;


import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;


/**
 * Created by agibsonccc on 9/12/14.
 * from : https://github.com/yusugomori/dl4j-0.4-examples
 * Iris : 벤치마크 데이터셋 (MNIST, LFW와 유사
 */
public class DBNIrisExample {

    private static Logger log = LoggerFactory.getLogger(DBNIrisExample.class);

    public static void main(String[] args) throws Exception {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        final int numRows = 4;
        final int numColumns = 1;   // 1차원
        int outputNum = 3;
        int numSamples = 150;       // 총 데이터 수
        int batchSize = 150;
        int iterations = 5;
        int splitTrainNum = (int) (batchSize * .8); // 학습 데이터 80%
        int seed = 123;
        int listenerFreq = 1;       // 매 학습 주기마다 기록.

        log.info("Load data....");
        //Iris 데이터를 읽어 온다.
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        
        //Data 포맷
        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();

        log.info("Split data....");
        //Dataset을 학습용과 시험용으로 구분
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();    //학습용
        DataSet test = testAndTrain.getTest();      //시험용
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Build model....");
        /*
        MultiLayerConfiguration conf = new MultiLayerConfiguration.Builder().layer().layer().....layer().build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        */
        //모델 구성을 정의
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            //전체 모델에 대한 설정
            .seed(seed) // Locks in weight initialization for tuning
            .iterations(iterations) // # training iterations predict/classify & backprop
            .learningRate(1e-6f) // Optimization step size
            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop to calculate gradients
            .l1(1e-1).regularization(true).l2(2e-4) //제어 규제? p.208~209 설명 참조
            .useDropConnect(true)   // 드롭아웃 (망 부분 생략) 허용
            .list() // # NN layers (doesn't count input layer) - 0.4 버전과 달라지 부분
          //개별 layer에 대한 설정      
          .layer(0 /* layer index */, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
            .nIn(numRows * numColumns) // # input nodes
            .nOut(4) // # fully connected hidden layer nodes. Add list if multiple layers.
            .weightInit(WeightInit.XAVIER) // Weight initialization
            .k(1) // # contrastive divergence iterations (대조 발산)
            .activation("relu") // Activation function type
            .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
            .updater(Updater.ADAGRAD)
            .dropOut(0.5)
            .build()
          ) // NN layer type
          .layer(1 /* layer index */, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
            .nIn(4) // # input nodes
            .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
            .weightInit(WeightInit.XAVIER) // Weight initialization
            .k(1) // # contrastive divergence iterations (대조 발산)
            .activation("relu") // Activation function type
            .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
            .updater(Updater.ADAGRAD)
            .dropOut(0.5)
            .build()
          ) // NN layer type      
          //출력 layer      
          .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
            .nIn(3) // # input nodes
            .nOut(outputNum) // # output nodes
            .activation("softmax")
            .build()
        ) // NN layer type
        .build();
        //모델 구축
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        //모델 초기화
        model.init();
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq),
//                new GradientPlotterIterationListener(listenerFreq),
//                new LossPlotterIterationListener(listenerFreq)));


        //로깅 설정
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        log.info("Train model (모델 학습) ....");
        //모델 학습
        model.fit(train);

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        log.info("Evaluate model (모델 평가)....");
        Evaluation eval = new Evaluation(outputNum);
        INDArray output = model.output(test.getFeatureMatrix());

        for (int i = 0; i < output.rows(); i++) {
            String actual = test.getLabels().getRow(i).toString().trim();
            String predicted = output.getRow(i).toString().trim();
            log.info("actual " + actual + " vs predicted " + predicted);
        }

        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

        OutputStream fos = Files.newOutputStream(Paths.get("coefficients.bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File("conf.json"), model.getLayerWiseConfigurations().toJson());

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));
        DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParams(newParams);
        System.out.println("Original network params \n" + model.params());
        System.out.println(savedNetwork.params());



    }
}