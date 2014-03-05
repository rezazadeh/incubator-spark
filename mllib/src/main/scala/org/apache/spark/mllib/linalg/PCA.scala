/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.util._


/**
 * Class used to obtain principal components
 */
class PCA {
  private var k: Int = 1

  /**
   * Set the number of top-k principle components to return
   */
  def setK(k: Int): PCA = {
    this.k = k
    this
  }

   /**
   * Compute PCA using the current set parameters
   */
  def compute(matrix: DenseMatrix): DenseMatrix = {
    computePCA(matrix, k)
  }

  /**
  * Principal Component Analysis.
  * Computes the top k principal component coefficients for the m-by-n data matrix X.
  * Rows of X correspond to observations and columns correspond to variables. 
  * The coefficient matrix is n-by-k. Each column of coeff contains coefficients
  * for one principal component, and the columns are in descending 
  * order of component variance.
  * This function centers the data and uses the 
  * singular value decomposition (SVD) algorithm. 
  *
  * All input and output is expected in DenseMatrix format
  *
  * @param matrix dense matrix to perform pca on
  * @param k Recover k principal components
  * @return An nxk matrix of principal components
  */
  def computePCA(matrix: DenseMatrix, k: Int): DenseMatrix = {
    val m = matrix.m
    val n = matrix.n
    val sc = matrix.rows.sparkContext

    if (m <= 0 || n <= 0) {
      throw new IllegalArgumentException("Expecting a well-formed matrix")
    }

    // compute column sums and normalize matrix
    val colSums = sc.broadcast(matrix.rows.map(x => x.data).fold(new Array[Double](n)){
      (a, b) => for(i <- 0 until n) {
                  a(i) += b(i)
                }
      a
    }).value
    
    val data = matrix.rows.map{
      x => for(i <- 0 until n) {
        x.data(i) = (x.data(i) - colSums(i) / m) / Math.sqrt(n - 1)
      }
      x
    }           
   
    val normalizedMatrix = DenseMatrix(data, m, n)
 
    val retV = new SVD().setK(k).setComputeU(false)
                .compute(normalizedMatrix).V
    retV
  }
}


/**
 * Top-level methods for calling Principal Component Analysis
 * NOTE: All matrices are DenseMatrix format
 */
object PCA {
  def main(args: Array[String]) {
    if (args.length < 6) {
      println("Usage: PCA <master> <matrix_file> <m> <n> " +
              "<k> <output_coefficient_file>")
      System.exit(1)
    }

    val (master, inputFile, m, n, k, output_u) = 
      (args(0), args(1), args(2).toInt, args(3).toInt,
      args(4).toInt, args(5))
    
    val sc = new SparkContext(master, "PCA")
    
    val rawData = sc.textFile(inputFile)
    val data = rawData.map { line =>
      val parts = line.split(',')
      MatrixEntry(parts(0).toInt, parts(1).toInt, parts(2).toDouble)
    }

    val u = new PCA().computePCA(LAUtils.spToDense(SparseMatrix(data, m, n)), k)
    
    println("Computed " + k + " principal vectors")
    u.rows.saveAsTextFile(output_u)
    System.exit(0)
  }
}


