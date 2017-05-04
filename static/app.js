/**
 * Created by opopovyc on 5/3/17.
 */

var res;

function get_data_for_plot(){
    $.get('/get_data_for_plot', function(data){
        res = data;
        console.log(res);
    });
};

function data_retrieval(){
    console.log(res);
};





