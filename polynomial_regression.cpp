#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <random>


//["land size", "bathrooms", "bedrooms", "garages", "house size", "age"]

Eigen::MatrixXd create_dataset(){

    int data_points = 1000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> land_size(500,800);
    std::uniform_int_distribution<int> bathrooms(1,3);
    std::uniform_int_distribution<int> bedrooms(1,5);
    std::uniform_int_distribution<int> garages(1,2);
    std::uniform_int_distribution<int> house_size(300,470);
    std::uniform_int_distribution<int> age(5,30);
    std::uniform_real_distribution<double> bias(10000,100000);

    Eigen::MatrixXd mymatrix(data_points,7);
    
    for (int i = 0; i < data_points; i++){
        int land_s = land_size(gen); 
        int baths = bathrooms(gen);
        int beds = bedrooms(gen); 
        int garagez = garages(gen);
        int hse_size = house_size(gen);
        int agee = age(gen);
        int b = bias(gen); 
        double out = baths * 1000 + pow(land_s,2) + beds * 1000 + garagez * 1000 + pow(hse_size,2) + agee * 1000 + b;
        
        mymatrix(i,0) = land_s;
        mymatrix(i,1) = baths;
        mymatrix(i,2) = beds;
        mymatrix(i,3) = garagez;
        mymatrix(i,4) = hse_size;
        mymatrix(i,5) = agee;
        mymatrix(i,6) = out;
    }

    return mymatrix;
    
}


Eigen::MatrixXd initialise_weights(int cols){
    Eigen::MatrixXd weight_row = Eigen::MatrixXd::Random(1,cols);
    return weight_row;
}


Eigen::MatrixXd predict(Eigen::MatrixXd x, Eigen::MatrixXd weights){

    //second degree poly.
    Eigen::MatrixXd xx = x.array().square();
    Eigen::MatrixXd final_xx = xx * weights.transpose();

    return final_xx;
}


double calculate_mae (Eigen::MatrixXd y, Eigen::MatrixXd y_pred){
    Eigen::MatrixXd mean_absolute_error = (y - y_pred).array().square().sqrt();
    double mean_absolute_error_fin = mean_absolute_error.sum()/y_pred.rows();

    return mean_absolute_error_fin;

}



Eigen::MatrixXd calculate_gradients(Eigen::MatrixXd y, Eigen::MatrixXd p, Eigen::MatrixXd x){
    
     //mae = y-pred
    Eigen::MatrixXd last_error = (y - p);//1000 1

    Eigen::MatrixXd pred_layer_error = x; //1000 6

    Eigen::MatrixXd gradient = last_error.transpose() * pred_layer_error;

    return gradient;
}



Eigen::MatrixXd train(int epochs, Eigen::MatrixXd weights, Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd predicted_y){
    
    Eigen::MatrixXd pre_trained_weights = weights;
    double alpha_rate = 0.00000000000001;

    //
    for (int i = 0; i < epochs; i++){
        //print mae.
        Eigen::MatrixXd out = predict(x,pre_trained_weights);
        double mae = calculate_mae(y, out);
        std::cout << "itteration " << i << std::endl;
        std::cout << "-----------------" << std::endl;
        std::cout << "cost -> " << mae << std::endl;
        std::cout << "\n\n";

        //compute gradients
        Eigen::MatrixXd gradients = calculate_gradients(y,predicted_y,x);
        
        //adjust weights
        Eigen::MatrixXd penalty = gradients.array() * Eigen::MatrixXd::Constant(gradients.rows(),gradients.cols(),alpha_rate).array();
        
        pre_trained_weights += penalty;

    }

    return pre_trained_weights;
}


int main(){

    //create data
    Eigen::MatrixXd my_data = create_dataset();

    //create weights;
    Eigen::MatrixXd weights = initialise_weights(my_data.cols() - 1);
      
    //organise the data
    Eigen::MatrixXd x = my_data.leftCols(my_data.cols() - 1);

    Eigen::MatrixXd y = my_data.rightCols(1);

    //intial predict and check mse
    Eigen::MatrixXd predicted_y = predict(x, weights);

    double mae = calculate_mae(y,predicted_y);
    std::cout << "mae = " << mae;
    
    //train train algoritim;
    Eigen::MatrixXd optimised_weights = train(260,weights,x,y,predicted_y);
    

    return 0;
}
