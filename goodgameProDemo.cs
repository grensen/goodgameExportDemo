using System;
using System.IO;
using System.Linq;

namespace goodgameProDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start Rosenblatt demo");  

            DateTime elapsed = DateTime.Now;
            // gg|one feed forward production demo
            string path = @"C:\goodgameProDemo\";
            string loadedNetwork = @"networks\Rosenblatt_784_200_100_10.txt";

            // init
            int[] u = { }; // network array
            NetworkTransformStringToIntArray(File.ReadLines(path + loadedNetwork).First());   // grab the network line

            Console.WriteLine("Run neural network " + string.Join(",", u) + "\n");

            int layer = u.Length - 1, neuronLen, weightLen;
            for (int n = neuronLen = weightLen = 0; n < u.Length; n++)
                neuronLen += u[n];
            for (int n = 1; n < u.Length; n++)
                weightLen += u[n - 1] * u[n];

            float[] neuron = new float[neuronLen], weight = new float[weightLen];

            // load trained weights
            FileStream Readfiles = new FileStream(path + loadedNetwork, FileMode.Open, FileAccess.Read);
            string[] backup = File.ReadLines(path + loadedNetwork).ToArray();
            for (int n = 1; n < backup.Length; n++)
                weight[n - 1] = float.Parse(backup[n]);
            Readfiles.Close(); // don't forget to close!

            for (int t = 0, len = 60000; t < 2; t++, len = 10000)
            {               
                // load training or test data
                FileStream image = new FileStream(t != 0 ? path + @"t10k-images.idx3-ubyte" : path + @"train-images.idx3-ubyte", FileMode.Open);
                FileStream label = new FileStream(t != 0 ? path + @"t10k-labels.idx1-ubyte" : path + @"train-labels.idx1-ubyte", FileMode.Open);
                image.Seek(16, 0); label.Seek(8, 0);

                int correct = 0, all = 0; 
                for (int i = 0; i < len; i++) // run 
                {
                    for (int n = 0; n < 784; ++n)
                        neuron[n] = image.ReadByte() / 255.0f;

                    int target = label.ReadByte();
                    int prediction = FeedForward();

                    correct += target == prediction ? 1 : 0;
                    all++;
                }

                if(t == 0)
                    Console.WriteLine("Training test ends after 60000 samples");                
                else
                    Console.WriteLine("Testing test ends after 10000 samples");
                Console.WriteLine("Accuracy = " + (correct * 100.0 / all).ToString("F2") + "%");
                Console.WriteLine("Correct = " + correct + "   incorrect = " + (all - correct) + "\n");
            }
            Console.WriteLine("Time = " + (((TimeSpan)(DateTime.Now - elapsed)).TotalMilliseconds / 1000.0).ToString("F2"));
            Console.WriteLine("\nEnd demo");
            Console.ReadLine();

            // local functions
            int FeedForward()
            {
                int pred = -1;
                float maxOut = float.MinValue;
                for (int i = 0, j = u[0], t = 0, w = 0; i < layer; i++, t += u[i - 1], w += u[i] * u[i - 1]) // layer
                {
                    for (int k = 0, kEnd = u[i + 1], nEnd = t + u[i]; k < kEnd; k++, j++) // neuron
                    {
                        float net = 0;
                        for (int n = t, m = w + k; n < nEnd; n++, m += kEnd) // weight
                                net += neuron[n] * weight[m];

                        if (i == layer - 1) // output layer prepare for softmax
                        {
                            neuron[j] = net;
                            if (net > maxOut) // grab the maxout here
                            { maxOut = net; pred = k; }
                        }
                        else // hidden relu
                            neuron[j] = net > 0 ? net : 0; // relu activation
                    }//--- k ends  
                }
                return pred;
            }
            void NetworkTransformStringToIntArray(string str, string val = "")
            {
                char[] chars = str.ToCharArray();
                Array.Resize<int>(ref u, chars.Count(x => x == ',') + 1);  // resize neural network array
                for (int i = 0, m = 0; i < chars.Length; i++) // transform the network
                {
                    if (chars[i] != ',') val += chars[i];
                    if (chars[i] == ',' || chars.Length - 2 < i)
                    {
                        u[m++] = Convert.ToInt16(val); // set neurons each layer
                        val = ""; // clear value for next layer
                    }
                }
            } // core transform
        } // main
    } //
} // ns
