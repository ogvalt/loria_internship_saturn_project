/**
 * Created by opopovyc on 5/3/17.
 */

var res;

function get_message(){
    $.get('/get_text', function(data){
        res = data;
    });
};

function data_retrieval(){
    console.log(res);
};





