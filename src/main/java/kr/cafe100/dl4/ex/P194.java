package kr.cafe100.dl4.ex;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author hohoonlee
 */
public class P194 {
    public static void main(String[] args) {
        INDArray x = Nd4j.create(new double[]{1,2,3,4,5,6}, new int[]{3,2});
        System.out.println("x");
        System.out.println(x);
        System.out.println("x.add(1)");
        System.out.println(x.add(1));
        
        INDArray y = Nd4j.create(new double[]{6,5,4,3,2,1}, new int[]{3,2});
        System.out.println("x.add(y)");
        System.out.println(x.add(y));
        System.out.println("x.sub(y)");
        System.out.println(x.sub(y));
        System.out.println("x.mul(y)");
        System.out.println(x.mul(y));
        System.out.println("x.div(y)");
        System.out.println(x.div(y));
        
        
        System.out.println("x.addi(y)");
        System.out.println(x.addi(y));
        System.out.println("x.subi(y)");
        System.out.println(x.subi(y));
        System.out.println("x.muli(y)");
        System.out.println(x.muli(y));
        System.out.println("x.divi(y)");
        System.out.println(x.divi(y));
    }
}
