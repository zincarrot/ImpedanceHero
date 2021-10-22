function z=importzdata(workbookFile, sheetName, dataLines)
%IMPORTFILE Import data from a spreadsheet
%  [UNTITLED, UNTITLED1, UNTITLED2, UNTITLED3, UNTITLED4, UNTITLED5,
%  UNTITLED6, UNTITLED7, UNTITLED8, UNTITLED9, UNTITLED10, UNTITLED11,
%  UNTITLED12, UNTITLED13, UNTITLED14, UNTITLED15, UNTITLED16,
%  UNTITLED17, UNTITLED18, UNTITLED19, UNTITLED20, UNTITLED21,
%  UNTITLED22, UNTITLED23, UNTITLED24, UNTITLED25, UNTITLED26,
%  UNTITLED27, UNTITLED28, UNTITLED29, UNTITLED30, UNTITLED31,
%  UNTITLED32, UNTITLED33, UNTITLED34, UNTITLED35, UNTITLED36,
%  UNTITLED37, UNTITLED38, UNTITLED39, UNTITLED40, UNTITLED41,
%  UNTITLED42, UNTITLED43, UNTITLED44, UNTITLED45, UNTITLED46,
%  UNTITLED47, UNTITLED48, UNTITLED49, UNTITLED50, UNTITLED51,
%  UNTITLED52, UNTITLED53, UNTITLED54, UNTITLED55, UNTITLED56,
%  UNTITLED57, UNTITLED58, UNTITLED59, UNTITLED60, UNTITLED61,
%  UNTITLED62, UNTITLED63, UNTITLED64, UNTITLED65, UNTITLED66,
%  UNTITLED67, UNTITLED68, UNTITLED69, UNTITLED70, UNTITLED71,
%  UNTITLED72, UNTITLED73, UNTITLED74] = IMPORTFILE(FILE) reads data
%  from the first worksheet in the Microsoft Excel spreadsheet file
%  named FILE.  Returns the data as column vectors.
%
%  [UNTITLED, UNTITLED1, UNTITLED2, UNTITLED3, UNTITLED4, UNTITLED5,
%  UNTITLED6, UNTITLED7, UNTITLED8, UNTITLED9, UNTITLED10, UNTITLED11,
%  UNTITLED12, UNTITLED13, UNTITLED14, UNTITLED15, UNTITLED16,
%  UNTITLED17, UNTITLED18, UNTITLED19, UNTITLED20, UNTITLED21,
%  UNTITLED22, UNTITLED23, UNTITLED24, UNTITLED25, UNTITLED26,
%  UNTITLED27, UNTITLED28, UNTITLED29, UNTITLED30, UNTITLED31,
%  UNTITLED32, UNTITLED33, UNTITLED34, UNTITLED35, UNTITLED36,
%  UNTITLED37, UNTITLED38, UNTITLED39, UNTITLED40, UNTITLED41,
%  UNTITLED42, UNTITLED43, UNTITLED44, UNTITLED45, UNTITLED46,
%  UNTITLED47, UNTITLED48, UNTITLED49, UNTITLED50, UNTITLED51,
%  UNTITLED52, UNTITLED53, UNTITLED54, UNTITLED55, UNTITLED56,
%  UNTITLED57, UNTITLED58, UNTITLED59, UNTITLED60, UNTITLED61,
%  UNTITLED62, UNTITLED63, UNTITLED64, UNTITLED65, UNTITLED66,
%  UNTITLED67, UNTITLED68, UNTITLED69, UNTITLED70, UNTITLED71,
%  UNTITLED72, UNTITLED73, UNTITLED74] = IMPORTFILE(FILE, SHEET) reads
%  from the specified worksheet.
%
%  [UNTITLED, UNTITLED1, UNTITLED2, UNTITLED3, UNTITLED4, UNTITLED5,
%  UNTITLED6, UNTITLED7, UNTITLED8, UNTITLED9, UNTITLED10, UNTITLED11,
%  UNTITLED12, UNTITLED13, UNTITLED14, UNTITLED15, UNTITLED16,
%  UNTITLED17, UNTITLED18, UNTITLED19, UNTITLED20, UNTITLED21,
%  UNTITLED22, UNTITLED23, UNTITLED24, UNTITLED25, UNTITLED26,
%  UNTITLED27, UNTITLED28, UNTITLED29, UNTITLED30, UNTITLED31,
%  UNTITLED32, UNTITLED33, UNTITLED34, UNTITLED35, UNTITLED36,
%  UNTITLED37, UNTITLED38, UNTITLED39, UNTITLED40, UNTITLED41,
%  UNTITLED42, UNTITLED43, UNTITLED44, UNTITLED45, UNTITLED46,
%  UNTITLED47, UNTITLED48, UNTITLED49, UNTITLED50, UNTITLED51,
%  UNTITLED52, UNTITLED53, UNTITLED54, UNTITLED55, UNTITLED56,
%  UNTITLED57, UNTITLED58, UNTITLED59, UNTITLED60, UNTITLED61,
%  UNTITLED62, UNTITLED63, UNTITLED64, UNTITLED65, UNTITLED66,
%  UNTITLED67, UNTITLED68, UNTITLED69, UNTITLED70, UNTITLED71,
%  UNTITLED72, UNTITLED73, UNTITLED74] = IMPORTFILE(FILE, SHEET,
%  DATALINES) reads from the specified worksheet for the specified row
%  interval(s). Specify DATALINES as a positive scalar integer or a
%  N-by-2 array of positive scalar integers for dis-contiguous row
%  intervals.
%
%  Example:
%  [Untitled, Untitled1, Untitled2, Untitled3, Untitled4, Untitled5, Untitled6, Untitled7, Untitled8, Untitled9, Untitled10, Untitled11, Untitled12, Untitled13, Untitled14, Untitled15, Untitled16, Untitled17, Untitled18, Untitled19, Untitled20, Untitled21, Untitled22, Untitled23, Untitled24, Untitled25, Untitled26, Untitled27, Untitled28, Untitled29, Untitled30, Untitled31, Untitled32, Untitled33, Untitled34, Untitled35, Untitled36, Untitled37, Untitled38, Untitled39, Untitled40, Untitled41, Untitled42, Untitled43, Untitled44, Untitled45, Untitled46, Untitled47, Untitled48, Untitled49, Untitled50, Untitled51, Untitled52, Untitled53, Untitled54, Untitled55, Untitled56, Untitled57, Untitled58, Untitled59, Untitled60, Untitled61, Untitled62, Untitled63, Untitled64, Untitled65, Untitled66, Untitled67, Untitled68, Untitled69, Untitled70, Untitled71, Untitled72, Untitled73, Untitled74] = importfile("C:\Users\Administrator\Desktop\9.15 Cell Growth\20190915_Sensor0_Test_1.xlsx", "sheet1", [2, 56]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 19-Sep-2019 09:50:24

%% Input handling

% If no sheet is specified, read first sheet
if nargin == 1 || isempty(sheetName)
    sheetName = 1;
end

% If row start and end points are not specified, define defaults
if nargin <= 2
    dataLines = [2, 100];
end

%% Setup the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 75);

% Specify sheet and range
opts.Sheet = sheetName;
opts.DataRange = "A" + dataLines(1, 1) + ":BW" + dataLines(1, 2);

% Specify column names and types
opts.VariableNames = ["Untitled", "Untitled1", "Untitled2", "Untitled3", "Untitled4", "Untitled5", "Untitled6", "Untitled7", "Untitled8", "Untitled9", "Untitled10", "Untitled11", "Untitled12", "Untitled13", "Untitled14", "Untitled15", "Untitled16", "Untitled17", "Untitled18", "Untitled19", "Untitled20", "Untitled21", "Untitled22", "Untitled23", "Untitled24", "Untitled25", "Untitled26", "Untitled27", "Untitled28", "Untitled29", "Untitled30", "Untitled31", "Untitled32", "Untitled33", "Untitled34", "Untitled35", "Untitled36", "Untitled37", "Untitled38", "Untitled39", "Untitled40", "Untitled41", "Untitled42", "Untitled43", "Untitled44", "Untitled45", "Untitled46", "Untitled47", "Untitled48", "Untitled49", "Untitled50", "Untitled51", "Untitled52", "Untitled53", "Untitled54", "Untitled55", "Untitled56", "Untitled57", "Untitled58", "Untitled59", "Untitled60", "Untitled61", "Untitled62", "Untitled63", "Untitled64", "Untitled65", "Untitled66", "Untitled67", "Untitled68", "Untitled69", "Untitled70", "Untitled71", "Untitled72", "Untitled73", "Untitled74"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

opts.ImportErrorRule = "omitrow";
opts.MissingRule = "omitrow";

% Specify variable properties
opts = setvaropts(opts, ["Untitled", "Untitled1", "Untitled2", "Untitled3", "Untitled4", "Untitled5", "Untitled6", "Untitled7", "Untitled8", "Untitled9", "Untitled10", "Untitled11", "Untitled12", "Untitled13", "Untitled14", "Untitled15", "Untitled16", "Untitled17", "Untitled18", "Untitled19", "Untitled20", "Untitled21", "Untitled22", "Untitled23", "Untitled24", "Untitled25", "Untitled26", "Untitled27", "Untitled28", "Untitled29", "Untitled30", "Untitled31", "Untitled32", "Untitled33", "Untitled34", "Untitled35", "Untitled36", "Untitled37", "Untitled38", "Untitled39", "Untitled40", "Untitled41", "Untitled42", "Untitled43", "Untitled44", "Untitled45", "Untitled46", "Untitled47", "Untitled48", "Untitled49", "Untitled50", "Untitled51", "Untitled52", "Untitled53", "Untitled54", "Untitled55", "Untitled56", "Untitled57", "Untitled58", "Untitled59", "Untitled60", "Untitled61", "Untitled62", "Untitled63", "Untitled64", "Untitled65", "Untitled66", "Untitled67", "Untitled68", "Untitled69", "Untitled70", "Untitled71", "Untitled72", "Untitled73", "Untitled74"], "TreatAsMissing", '');


% Import the data
Sensor0Test28 = readtable(workbookFile, opts, "UseExcel", false);

for idx = 2:size(dataLines, 1)
    opts.DataRange = "A" + dataLines(idx, 1) + ":BW" + dataLines(idx, 2);
    tb = readtable(workbookFile, opts, "UseExcel", false);
    Sensor0Test28 = [Sensor0Test28; tb]; %#ok<AGROW>
end

v=table2array(Sensor0Test28);
z=zeros(1,15);
for k=1:1:15
zreal=v(:,k*5+-2);
zimag=v(:,k*5+-1);
zreal=zreal(zreal<1e30);
zimag=zimag(zimag<1e30);
z(k)=mean(nonzeros(zreal))+1i*mean(nonzeros(zimag));
end
end